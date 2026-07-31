"""Replay a recorded camera corpus — the benchmark's input, and the reason the
benchmarks need no simulator.

A corpus is a directory:

    meta.json       one document: site, resolution, encoding, provenance
    manifest.jsonl  one line per frame set: timestamps + ground truth
    frames.zip      the frames, as JPEG, named "<camera>/<index>.jpg"

WHY THIS SHAPE. A single zip ships in git and stays inspectable with ordinary
tools; JSON Lines streams and survives a truncated write, so an interrupted
recording is still a usable corpus. Nothing is pickled, so a corpus recorded today
is readable by any future version and on any architecture.

WHY JPEG, GIVEN IT IS LOSSY. The corpus *is* the benchmark input, so what matters
is that every backend sees identical bytes, not that those bytes match the
simulator's framebuffer exactly. Compressed frames are also what a real deployment
receives, since the camera transport encodes them anyway. The quality setting is
recorded in `meta.json` so the input is fully described.

DECODE COST IS A CHOICE, NOT AN ACCIDENT. On an Arm CPU, JPEG decode is a real
part of the per-frame budget. `mode` decides whether it lands inside the measured
region:

    "decoded"  decode everything up front; steps return ready arrays. Measures the
               pipeline alone. The default, because it isolates what is being
               optimized.
    "encoded"  hold JPEG bytes in memory, decode on each step. Models the
               deployment path, where frames arrive compressed.
    "stream"   read from the zip on each step. Lowest memory, adds I/O noise;
               for corpora too large to hold.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from dronevision.io.sources.base import FrameSource

CORPUS_VERSION = 1
META_FILE = "meta.json"
MANIFEST_FILE = "manifest.jsonl"
FRAMES_FILE = "frames.zip"

MODES = ("decoded", "encoded", "stream")


def frame_name(cam, index):
    """Path of one frame inside the zip. Zero-padded so listings sort correctly."""
    return f"{cam}/{index:06d}.jpg"


def read_meta(path):
    with open(Path(path) / META_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_manifest(path):
    """Frame-set records, in order. Tolerates a truncated final line."""
    records = []
    with open(Path(path) / MANIFEST_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Only ever the last line, from a recording cut mid-write.
                break
    return records


class ReplaySource(FrameSource):
    """A `FrameSource` over a recorded corpus, with a cursor.

    Reads like a live source — `latest(cam)` and `meta(cam)` return the frame set
    at the cursor — so a pipeline cannot tell the difference. `step()` advances.

        src = ReplaySource("data/corpus")
        while src.step():
            est = pipeline.locate_from(src)
            error = np.linalg.norm(est.position - src.truth)
    """

    def __init__(self, path, mode="decoded", cams=None, loop=False,
                 start=0, limit=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.path = Path(path)
        if not (self.path / META_FILE).exists():
            raise FileNotFoundError(
                f"{self.path} is not a corpus (no {META_FILE}). "
                f"Record one with bridge/recorder.py")

        self.meta_doc = read_meta(self.path)
        version = self.meta_doc.get("corpus_version")
        if version != CORPUS_VERSION:
            raise ValueError(
                f"corpus version {version} but this build reads v{CORPUS_VERSION}")

        records = read_manifest(self.path)
        if start:
            records = records[start:]
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f"{self.path}: no frame sets in range")
        self._records = records

        self.cams = list(cams) if cams else list(self.meta_doc["cameras"])
        self.mode = mode
        self.loop = loop
        self._i = -1                      # before the first frame set
        self._zip = None
        self._cache = {}                  # (cam, index) -> ndarray or bytes

        self._zip = zipfile.ZipFile(self.path / FRAMES_FILE, "r")
        if mode in ("decoded", "encoded"):
            self._preload(decode=(mode == "decoded"))
            self._zip.close()
            self._zip = None

    # -- loading ------------------------------------------------------------

    def _preload(self, decode):
        for rec in self._records:
            for cam in self.cams:
                entry = rec["cams"].get(cam)
                if not entry:
                    continue
                raw = self._zip.read(entry["f"])
                self._cache[(cam, rec["i"])] = self._decode(raw) if decode else raw

    @staticmethod
    def _decode(raw):
        """JPEG bytes -> RGB uint8 HxWx3, matching what a live source delivers."""
        import cv2
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("corrupt JPEG in corpus")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -- cursor -------------------------------------------------------------

    def __len__(self):
        return len(self._records)

    @property
    def index(self):
        """Cursor position, or -1 before the first `step()`."""
        return self._i

    @property
    def record(self):
        return None if self._i < 0 else self._records[self._i]

    def step(self):
        """Advance one frame set. False when exhausted (unless looping)."""
        if self._i + 1 >= len(self._records):
            if not self.loop:
                return False
            self._i = 0
            return True
        self._i += 1
        return True

    def seek(self, index):
        if not -1 <= index < len(self._records):
            raise IndexError(f"index {index} outside 0..{len(self._records) - 1}")
        self._i = index

    def rewind(self):
        self._i = -1

    def __iter__(self):
        """Iterate frame sets, yielding self so `locate_from(src)` keeps working."""
        self.rewind()
        while self.step():
            yield self

    # -- FrameSource --------------------------------------------------------

    def latest(self, cam):
        rec = self.record
        if rec is None:
            return None
        entry = rec["cams"].get(cam)
        if not entry:
            return None
        key = (cam, rec["i"])
        if self.mode == "decoded":
            return self._cache.get(key)
        if self.mode == "encoded":
            raw = self._cache.get(key)
            return None if raw is None else self._decode(raw)
        return self._decode(self._zip.read(entry["f"]))

    def meta(self, cam):
        rec = self.record
        if rec is None:
            return None
        entry = rec["cams"].get(cam)
        if not entry:
            return None
        return {"stamp": entry.get("stamp"), "seq": entry.get("seq"),
                "frame_id": cam}

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    # -- ground truth -------------------------------------------------------

    @property
    def truth(self):
        """Recorded world position of the vehicle at the cursor, or None."""
        rec = self.record
        if rec is None or rec.get("truth") is None:
            return None
        return np.array(rec["truth"], float)

    @property
    def truth_speed(self):
        """Speed at the cursor from neighbouring truth samples (m/s), or None.

        Lets a benchmark separate hovering accuracy from accuracy under motion,
        which are different numbers and get conflated otherwise.
        """
        if self._i < 1:
            return None
        a, b = self._records[self._i - 1], self._records[self._i]
        if a.get("truth") is None or b.get("truth") is None:
            return None
        dt = b["t"] - a["t"]
        if dt <= 0:
            return None
        return float(np.linalg.norm(np.array(b["truth"]) - np.array(a["truth"])) / dt)

    @property
    def duration(self):
        return self._records[-1]["t"] - self._records[0]["t"]

    def summary(self):
        truths = [r["truth"] for r in self._records if r.get("truth")]
        out = {
            "path": str(self.path),
            "frame_sets": len(self._records),
            "cameras": self.cams,
            "duration_s": round(self.duration, 2),
            "mode": self.mode,
            "site": self.meta_doc.get("site"),
            "resolution": self.meta_doc.get("resolution"),
            "jpeg_quality": self.meta_doc.get("jpeg_quality"),
        }
        if truths:
            a = np.array(truths)
            out["truth_bounds"] = {
                "min": [round(v, 3) for v in a.min(0)],
                "max": [round(v, 3) for v in a.max(0)],
            }
        return out
