"""Corpus replay tests.

Builds a tiny synthetic corpus in a temp directory, so these run without the
recorded one and without a simulator. If the shipped corpus is present, a few
extra tests check it for real.

The corpus is the measurement instrument for every later claim, so what is tested
here is mostly that it cannot lie quietly: version mismatches raise, truncated
manifests degrade instead of corrupting, and the three decode modes agree.
"""
import json
import zipfile

import cv2
import numpy as np
import pytest

from dronevision.io.sources.replay import (
    CORPUS_VERSION, FRAMES_FILE, MANIFEST_FILE, META_FILE, MODES, ReplaySource,
    frame_name, read_manifest, read_meta,
)

CAMS = ["cam_ne", "cam_nw", "cam_sw", "cam_se"]


def make_corpus(path, n=6, cams=CAMS, quality=95, truth=True, version=None):
    """A minimal but valid corpus: each frame has a blob at a known pixel."""
    path.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(path / FRAMES_FILE, "w", zipfile.ZIP_STORED) as zf:
        for i in range(n):
            rec = {"i": i, "t": 100.0 + i * 0.05,
                   "truth": [0.1 * i, -0.05 * i, 2.0 + 0.01 * i] if truth else None,
                   "cams": {}}
            for c in cams:
                img = np.zeros((360, 640, 3), np.uint8)
                img[:] = (40, 45, 50)
                cv2.circle(img, (100 + 5 * i, 150), 8, (0, 0, 255), -1)   # BGR red
                ok, buf = cv2.imencode(".jpg", img,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                assert ok
                name = frame_name(c, i)
                zf.writestr(name, buf.tobytes())
                rec["cams"][c] = {"f": name, "stamp": rec["t"], "seq": i}
            manifest.append(rec)

    with open(path / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        for rec in manifest:
            fh.write(json.dumps(rec) + "\n")
    with open(path / META_FILE, "w", encoding="utf-8") as fh:
        json.dump({"corpus_version": CORPUS_VERSION if version is None else version,
                   "site": "factory", "cameras": cams, "resolution": [640, 360],
                   "jpeg_quality": quality, "frame_sets": n}, fh)
    return path


@pytest.fixture
def corpus(tmp_path):
    return make_corpus(tmp_path / "corpus")


# --------------------------------------------------------------------------
# Cursor semantics
# --------------------------------------------------------------------------

def test_starts_before_the_first_frame(corpus):
    src = ReplaySource(corpus)
    assert src.index == -1
    assert src.record is None
    # Reading before stepping must yield nothing rather than frame 0 — otherwise a
    # loop that forgets to step silently measures the same frame forever.
    assert src.latest("cam_ne") is None
    assert src.truth is None


def test_steps_through_every_frame_once(corpus):
    src = ReplaySource(corpus)
    seen = []
    while src.step():
        seen.append(src.index)
    assert seen == list(range(len(src)))
    assert src.step() is False


def test_iteration_rewinds_and_yields_self(corpus):
    src = ReplaySource(corpus)
    first = [s.index for s in src]
    second = [s.index for s in src]
    assert first == second == list(range(len(src)))


def test_loop_wraps(corpus):
    src = ReplaySource(corpus, loop=True)
    idx = [src.index for _ in range(len(src) + 2) if src.step()]
    assert idx[-1] < idx[-3]              # wrapped rather than stopped


def test_seek_and_rewind(corpus):
    src = ReplaySource(corpus)
    src.seek(3)
    assert src.index == 3 and src.record["i"] == 3
    src.rewind()
    assert src.index == -1
    with pytest.raises(IndexError):
        src.seek(len(src))


def test_start_and_limit(corpus):
    src = ReplaySource(corpus, start=2, limit=3)
    assert len(src) == 3
    src.step()
    assert src.record["i"] == 2


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_all_modes_return_equivalent_frames(corpus, mode):
    src = ReplaySource(corpus, mode=mode)
    src.step()
    img = src.latest("cam_ne")
    assert img is not None
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    src.close()


def test_modes_agree_pixel_for_pixel(corpus):
    """The mode only decides *when* decoding happens, never what comes out."""
    refs = {}
    for mode in MODES:
        src = ReplaySource(corpus, mode=mode)
        src.step()
        refs[mode] = src.latest("cam_sw").copy()
        src.close()
    for mode in MODES[1:]:
        np.testing.assert_array_equal(refs[MODES[0]], refs[mode])


def test_frames_are_rgb_not_bgr(corpus):
    """The synthetic blob is red; a BGR/RGB slip would make it blue and every
    colour detector would silently stop finding anything."""
    src = ReplaySource(corpus)
    src.step()
    img = src.latest("cam_ne")
    r, g, b = img[150, 100]
    assert r > 150 and b < 100, f"expected red, got RGB=({r},{g},{b})"


def test_all_returns_every_camera(corpus):
    src = ReplaySource(corpus)
    src.step()
    frames = src.all()
    assert set(frames) == set(CAMS)
    assert all(v is not None for v in frames.values())


def test_camera_subset(corpus):
    src = ReplaySource(corpus, cams=["cam_ne", "cam_sw"])
    src.step()
    assert set(src.all()) == {"cam_ne", "cam_sw"}


def test_meta_carries_stamps(corpus):
    src = ReplaySource(corpus)
    src.step()
    m = src.meta("cam_ne")
    assert m["stamp"] == pytest.approx(100.0)
    assert m["seq"] == 0
    assert src.meta("nonexistent") is None


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def test_truth_tracks_the_cursor(corpus):
    src = ReplaySource(corpus)
    src.step()
    np.testing.assert_allclose(src.truth, [0.0, 0.0, 2.0])
    src.step()
    np.testing.assert_allclose(src.truth, [0.1, -0.05, 2.01])


def test_truth_speed_needs_two_samples(corpus):
    src = ReplaySource(corpus)
    src.step()
    assert src.truth_speed is None        # nothing to difference against yet
    src.step()
    expected = np.linalg.norm([0.1, -0.05, 0.01]) / 0.05
    assert src.truth_speed == pytest.approx(expected, rel=1e-6)


def test_corpus_without_truth(tmp_path):
    src = ReplaySource(make_corpus(tmp_path / "c", truth=False))
    src.step()
    assert src.truth is None
    assert src.truth_speed is None


def test_summary_reports_bounds(corpus):
    s = ReplaySource(corpus).summary()
    assert s["frame_sets"] == 6
    assert s["cameras"] == CAMS
    assert s["duration_s"] == pytest.approx(0.25)
    assert "truth_bounds" in s


# --------------------------------------------------------------------------
# Refusing to be silently wrong
# --------------------------------------------------------------------------

def test_missing_corpus_names_the_recorder(tmp_path):
    with pytest.raises(FileNotFoundError, match="recorder"):
        ReplaySource(tmp_path / "nope")


def test_version_mismatch_refuses(tmp_path):
    c = make_corpus(tmp_path / "c", version=CORPUS_VERSION + 99)
    with pytest.raises(ValueError, match="corpus version"):
        ReplaySource(c)


def test_unknown_mode_refuses(corpus):
    with pytest.raises(ValueError, match="unknown mode"):
        ReplaySource(corpus, mode="magic")


def test_empty_range_refuses(corpus):
    with pytest.raises(ValueError, match="no frame sets"):
        ReplaySource(corpus, start=999)


def test_truncated_manifest_degrades_gracefully(tmp_path):
    """A recording killed mid-write must still yield a usable corpus, because the
    alternative is discarding a flight that cannot be repeated."""
    c = make_corpus(tmp_path / "c", n=6)
    text = (c / MANIFEST_FILE).read_text(encoding="utf-8")
    lines = text.splitlines()
    truncated = "\n".join(lines[:-1]) + "\n" + lines[-1][:20]   # cut mid-JSON
    (c / MANIFEST_FILE).write_text(truncated, encoding="utf-8")

    assert len(read_manifest(c)) == 5
    assert len(ReplaySource(c)) == 5


def test_read_helpers(corpus):
    assert read_meta(corpus)["site"] == "factory"
    assert len(read_manifest(corpus)) == 6


# --------------------------------------------------------------------------
# Against the shipped corpus, when present
# --------------------------------------------------------------------------

def _shipped():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "corpus"
    return p if (p / META_FILE).exists() else None


@pytest.mark.skipif(_shipped() is None, reason="no recorded corpus in data/corpus")
def test_shipped_corpus_is_readable_and_localizable():
    from dronevision.l5_estimation.smoothing import EMASmoother
    from dronevision.l2_perception.detector import ColorDetector
    from dronevision.pipeline import LocalizationPipeline
    from dronevision.l4_triangulation.calibration import load_site

    src = ReplaySource(_shipped())
    assert len(src) > 50, "corpus too short to mean anything"

    pipe = LocalizationPipeline(cal=load_site("factory"), detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    errs = []
    for s in src:
        est = pipe.locate_from(s)
        if est is not None and s.truth is not None:
            errs.append(float(np.linalg.norm(np.array(est.position) - s.truth)))

    assert len(errs) > 0.9 * len(src), "pipeline failed on many frame sets"
    mean_mm = np.mean(errs) * 1000
    # Loose on purpose: this guards against a regression that breaks localization
    # outright, not against small accuracy drift, which belongs in a benchmark.
    assert mean_mm < 50, f"mean error {mean_mm:.1f} mm — localization is broken"


@pytest.mark.skipif(_shipped() is None, reason="no recorded corpus in data/corpus")
def test_shipped_corpus_records_its_capture_conditions():
    """Provenance the benchmark needs: without these a number is not comparable."""
    m = read_meta(_shipped())
    for key in ("site", "cameras", "resolution", "jpeg_quality", "marker_dz",
                "truth_pose", "site_config"):
        assert key in m, f"meta.json is missing {key!r}"
