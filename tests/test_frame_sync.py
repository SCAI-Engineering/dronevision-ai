"""FrameSync must actually enforce its tolerance, and the pipeline must use it.

This exists because of a real, measured failure mode, not a hypothetical one: a
corpus recorded at 2 m/s with hard cornering showed camera capture stamps up to
84 ms apart within a single "simultaneous" frame set (cameras share no clock even
in simulation), and roughly 150 mm of the resulting 3D error traced directly to
that skew. `FrameSync` existed as a scaffold before this — buffering frames but
returning them regardless of alignment. These tests pin the behaviour that
actually discards a lagging camera instead of triangulating it as current.
"""
import numpy as np
import pytest

from dronevision.l1_image_sync.aligner import FrameSync

CAMS = ["cam_a", "cam_b", "cam_c"]
FRAME = np.zeros((4, 4, 3), np.uint8)


@pytest.fixture
def sync():
    return FrameSync(CAMS, tol_ms=50.0)


def test_a_single_camera_is_always_aligned_with_itself(sync):
    sync.add("cam_a", FRAME, stamp=1.000)
    assert set(sync.latest_aligned()) == {"cam_a"}


def test_cameras_within_tolerance_are_all_kept(sync):
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=1.030)   # 30 ms behind, tol is 50 ms
    sync.add("cam_c", FRAME, stamp=1.010)   # 10 ms behind
    assert set(sync.latest_aligned()) == {"cam_a", "cam_b", "cam_c"}


def test_a_camera_that_fell_behind_is_dropped_not_triangulated_stale():
    """The measured case: one camera lags past the tolerance while the rest are
    tight together. It must be excluded, not averaged in as if current."""
    sync = FrameSync(CAMS, tol_ms=50.0)
    sync.add("cam_a", FRAME, stamp=1.084)   # newest
    sync.add("cam_b", FRAME, stamp=1.080)   # 4 ms behind newest -> kept
    sync.add("cam_c", FRAME, stamp=1.000)   # 84 ms behind newest -> dropped
    aligned = sync.latest_aligned()
    assert set(aligned) == {"cam_a", "cam_b"}
    assert "cam_c" not in aligned


def test_with_only_two_cameras_a_lone_straggler_leaves_the_other_alone(sync):
    """With just two buffered cameras there is no cluster to fall back on if they
    disagree -- the newest one is returned by itself, same as a single camera."""
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=1.020)
    assert set(sync.latest_aligned()) == {"cam_a", "cam_b"}
    sync.add("cam_b", FRAME, stamp=1.200)   # cam_b jumps far ahead, no pair aligns
    assert set(sync.latest_aligned()) == {"cam_b"}


def test_an_isolated_newest_frame_does_not_discard_an_aligned_majority():
    """The bug this file exists to prevent, reproduced directly: three cameras
    land within a few ms of each other while a fourth is newer but alone. Picking
    the largest *aligned cluster* must keep the three, not collapse to the one
    just because it is freshest. Measured on a real corpus: anchoring to the
    single newest frame found >=2 usable cameras in 65% of frames; searching for
    the best cluster found one in 100% of the same frames."""
    sync = FrameSync(CAMS + ["cam_d"], tol_ms=50.0)
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=1.010)
    sync.add("cam_c", FRAME, stamp=1.005)
    sync.add("cam_d", FRAME, stamp=1.150)   # newest, but isolated
    assert set(sync.latest_aligned()) == {"cam_a", "cam_b", "cam_c"}


def test_the_tighter_of_two_equally_large_clusters_wins():
    sync = FrameSync(CAMS + ["cam_d"], tol_ms=50.0)
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=1.002)    # {a, b}: 2 ms apart
    sync.add("cam_c", FRAME, stamp=1.100)
    sync.add("cam_d", FRAME, stamp=1.140)    # {c, d}: 40 ms apart, still within tol
    assert set(sync.latest_aligned()) == {"cam_a", "cam_b"}


def test_a_frame_with_no_timestamp_cannot_be_judged_so_it_is_dropped(sync):
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=None)
    assert set(sync.latest_aligned()) == {"cam_a"}


def test_an_empty_buffer_aligns_to_nothing(sync):
    assert sync.latest_aligned() == {}


def test_spread_ms_reports_the_full_spread_not_the_aligned_subset(sync):
    """spread_ms is the 'how much is this mattering' diagnostic -- it must not
    quietly agree with latest_aligned, or it stops being useful as one."""
    sync.add("cam_a", FRAME, stamp=1.000)
    sync.add("cam_b", FRAME, stamp=1.084)
    assert sync.spread_ms() == pytest.approx(84.0)
    assert set(sync.latest_aligned()) == {"cam_b"}   # cam_a excluded at tol=50ms


def test_stamp_reports_what_latest_aligned_actually_used(sync):
    sync.add("cam_a", FRAME, stamp=1.234)
    assert sync.stamp("cam_a") == pytest.approx(1.234)
    assert sync.stamp("cam_never_added") is None


# ---------------------------------------------------------------------------
# Wired into the pipeline
# ---------------------------------------------------------------------------

class _FakeSource:
    """Minimal `FrameSource`: fixed per-camera frame/stamp, no transport."""

    clock = "sim"

    def __init__(self, stamps):
        self._stamps = stamps

    def latest(self, cam):
        return FRAME if cam in self._stamps else None

    def meta(self, cam):
        if cam not in self._stamps:
            return None
        return {"stamp": self._stamps[cam], "seq": 1, "frame_id": cam}


def test_pipeline_without_sync_uses_every_camera_regardless_of_skew():
    """Default behaviour, unchanged: `sync=None` is today's pipeline exactly."""
    from dronevision.l2_perception.detector import ColorDetector
    from dronevision.l4_triangulation.calibration import load_site
    from dronevision.pipeline import LocalizationPipeline

    site = load_site("factory")
    src = _FakeSource({c: float(i) * 0.1 for i, c in enumerate(site.cam_names)})
    pipe = LocalizationPipeline(cal=site, detector=ColorDetector())
    assert pipe.sync is None
    # ColorDetector finds nothing on an all-zero frame, so this only checks that
    # acquisition itself is unfiltered -- every active camera was offered.
    frames = {c: src.latest(c) for c in pipe.active_cams}
    assert set(frames) == set(site.cam_names)


def test_pipeline_with_sync_drops_the_lagging_camera_before_detection():
    """The integration point: a camera outside tolerance must never reach
    `detect()`, not just be flagged somewhere downstream."""
    from dronevision.l4_triangulation.calibration import load_site
    from dronevision.pipeline import LocalizationPipeline

    site = load_site("factory")
    cams = site.cam_names
    stamps = {c: 1.084 for c in cams}
    stamps[cams[0]] = 1.000   # one camera 84 ms behind the rest

    seen = []

    class _RecordingDetector:
        target_offset_key = None
        name = "recording"

        def detect(self, img, cam=None):
            seen.append(cam)
            return None

    src = _FakeSource(stamps)
    sync = FrameSync(cams, tol_ms=50.0)
    pipe = LocalizationPipeline(cal=site, detector=_RecordingDetector(), sync=sync)

    pipe.locate_from(src)

    assert cams[0] not in seen, "the lagging camera reached the detector"
    assert set(seen) == set(cams[1:])
