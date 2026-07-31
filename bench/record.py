#!/usr/bin/env python3
"""Record a corpus from the camera service.

Consumes the same published service the AI consumes, so the recorded frames are
exactly the frames the AI would have seen — including the encoding and the network
timing. No simulator API is touched.

Once a corpus exists, every accuracy and throughput claim is reproducible on any
machine with no camera service running at all, which is what makes benchmarking on
a Raspberry Pi (or by anyone else) possible.

    python -m bench.record --seconds 30 --out data/corpus

FLY THE VEHICLE WHILE THIS RUNS. A stationary corpus measures almost nothing: no
tracking, no coasting, no motion-dependent detector behaviour, and every position
error is the same error.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from dronevision.io.sources.net import NetSource
from dronevision.io.sources.replay import (
    CORPUS_VERSION, FRAMES_FILE, MANIFEST_FILE, META_FILE, frame_name,
)
from dronevision.l4_triangulation.calibration import load_site


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--frames", default="tcp://127.0.0.1:5555")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--quality", type=int, default=90,
                    help="re-encode quality; below 80 costs localization accuracy")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    site = load_site(a.site)
    out = Path(a.out)
    if (out / META_FILE).exists() and not a.force:
        raise SystemExit(f"{out} already holds a corpus; pass --force to replace it")
    out.mkdir(parents=True, exist_ok=True)

    src = NetSource(site.cam_names, endpoint=a.frames)
    print(f"[rec] waiting for cameras on {a.frames} ...", flush=True)
    if not src.wait_ready(timeout=30):
        raise SystemExit(
            f"no frames from {src.missing()}. Start the camera service on the "
            f"simulator host:\n  tools/service/frame_publisher.py --bind tcp://0.0.0.0:5555")

    print(f"[rec] recording {a.seconds:.0f}s at {a.rate:.0f} Hz -> {out}")
    print("[rec] FLY THE VEHICLE NOW — a stationary corpus measures almost nothing\n")

    period = 1.0 / a.rate
    deadline = time.time() + a.seconds
    manifest, n_bytes, skipped = [], 0, 0
    last_seq = {}
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), a.quality]

    with zipfile.ZipFile(out / FRAMES_FILE, "w", zipfile.ZIP_STORED) as zf:
        # ZIP_STORED: the payload is already JPEG, so deflating costs CPU and
        # saves nothing, and stored entries read back faster.
        i = 0
        while time.time() < deadline:
            t0 = time.time()
            frames = {c: src.latest(c) for c in site.cam_names}
            metas = {c: src.meta(c) for c in site.cam_names}
            have = [c for c in site.cam_names
                    if frames[c] is not None and metas[c] is not None]

            # EVERY camera must have renewed its frame. "Any camera is new" pairs
            # one fresh frame with stale ones, inflating the count with sets that
            # carry no new information and are inconsistent in time.
            if (len(have) < site.min_views
                    or not all(metas[c]["seq"] != last_seq.get(c) for c in have)):
                skipped += 1
                time.sleep(max(0.0, period - (time.time() - t0)))
                continue
            for c in have:
                last_seq[c] = metas[c]["seq"]

            truth = src.truth
            rec = {"i": i,
                   "t": round(min(metas[c]["stamp"] for c in have), 6),
                   "truth": None if truth is None
                            else [round(float(v), 5) for v in truth],
                   "cams": {}}
            for c in have:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frames[c], cv2.COLOR_RGB2BGR), enc)
                if not ok:
                    continue
                name = frame_name(c, i)
                zf.writestr(name, buf.tobytes())
                n_bytes += len(buf.tobytes())
                rec["cams"][c] = {"f": name, "stamp": round(metas[c]["stamp"], 6),
                                  "seq": metas[c]["seq"]}
            manifest.append(rec)
            i += 1
            if i % 50 == 0:
                tz = "" if rec["truth"] is None else "  z=%+.2f" % rec["truth"][2]
                print("  %4d sets  %5.1f MB%s" % (i, n_bytes / 1e6, tz), flush=True)
            time.sleep(max(0.0, period - (time.time() - t0)))

    src.close()
    if not manifest:
        raise SystemExit("recorded nothing — was the camera service publishing?")

    with open(out / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        for rec in manifest:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    first = manifest[0]["cams"]
    any_cam = next(iter(first))
    truths = [r["truth"] for r in manifest if r["truth"]]
    meta = {
        "corpus_version": CORPUS_VERSION,
        "site": site.name,
        "cameras": [c for c in site.cam_names if c in first],
        "resolution": site[any_cam].resolution,
        "jpeg_quality": a.quality,
        "frame_sets": len(manifest),
        "requested_rate_hz": a.rate,
        "duration_s": round(manifest[-1]["t"] - manifest[0]["t"], 3),
        "marker_dz": site.marker_dz,
        "source": a.frames,
        "note": a.note,
        # Stored so a corpus can be matched to the geometry it was captured with:
        # a later calibration change invalidates the recorded ground truth.
        "site_config": site.raw,
    }
    if truths:
        arr = np.array(truths)
        meta["truth_bounds"] = {"min": [round(float(v), 3) for v in arr.min(0)],
                                "max": [round(float(v), 3) for v in arr.max(0)]}
    # How far apart in time a single set's cameras actually are. This bounds
    # achievable accuracy independently of the detector, so a benchmark that omits
    # it will blame the pipeline for the capture.
    spreads = np.array([(max(c["stamp"] for c in r["cams"].values())
                         - min(c["stamp"] for c in r["cams"].values())) * 1000.0
                        for r in manifest if len(r["cams"]) > 1])
    if spreads.size:
        meta["camera_skew_ms"] = {"mean": round(float(spreads.mean()), 2),
                                  "p95": round(float(np.percentile(spreads, 95)), 2),
                                  "max": round(float(spreads.max()), 2)}

    with open(out / META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n[rec] {len(manifest)} sets, {n_bytes / 1e6:.1f} MB -> {out}")
    print(f"      {meta['duration_s']:.1f}s of capture, skipped {skipped} stale/short")
    if spreads.size:
        print(f"      camera skew: mean {meta['camera_skew_ms']['mean']:.1f} ms, "
              f"max {meta['camera_skew_ms']['max']:.1f} ms")
    if truths:
        b = meta["truth_bounds"]
        print(f"      truth bounds  min {b['min']}  max {b['max']}")
        if (np.array(b["max"]) - np.array(b["min"])).max() < 0.25:
            print("      !! the vehicle barely moved — this corpus will not exercise")
            print("         tracking, coasting, or motion-dependent detectors")
    else:
        print("      !! no ground truth published — accuracy cannot be measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
