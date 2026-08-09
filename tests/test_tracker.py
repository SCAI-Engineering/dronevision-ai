"""`TrackedDetector` must report the network cost it actually causes.

Exists because `TrackedCamera.detect_rate` undercounts on its own: its
`detections` counter only increments on a *successful* detect, so a benchmark
built on it would understate cost exactly on the frames where the target is hard
to find -- the frames that matter most. `TrackedDetector` wraps the underlying
detector with its own call counter instead of trusting the tracker's.
"""
import numpy as np
import pytest

from dronevision.l3_association.tracker import TrackedCamera, TrackedDetector

CAMS = ["cam_a", "cam_b"]
FRAME = np.zeros((4, 4, 3), np.uint8)


class _FakeDetector:
    """Always finds the target at a fixed pixel; counts its own calls too, as
    an independent check on `TrackedDetector`'s counter."""

    name = "fake"
    target_offset_key = None

    def __init__(self):
        self.calls = 0

    def detect(self, img, cam=None):
        self.calls += 1
        return (10.0, 20.0)


class _NeverFindsDetector(_FakeDetector):
    def detect(self, img, cam=None):
        self.calls += 1
        return None


def test_wraps_the_detector_name_and_offset_key():
    inner = _FakeDetector()
    inner.target_offset_key = "airframe_dz"
    td = TrackedDetector(inner, CAMS)
    assert td.name == "tracked/fake"
    assert td.target_offset_key == "airframe_dz"


def test_calls_counts_every_network_invocation_not_just_successes():
    """The gap `TrackedCamera.detect_rate` has, closed: a detector that never
    finds anything still runs the network on schedule, and that must count."""
    inner = _NeverFindsDetector()
    td = TrackedDetector(inner, CAMS, detect_every=5)
    for _ in range(10):
        td.detect(FRAME, cam="cam_a")
    # With no track ever acquired, `step()` forces a detect on every tick
    # (`not have_track`) -- worth pinning, since it is the worst case for cost.
    assert td.calls == 10
    assert inner.calls == 10


def test_calls_reflects_the_schedule_once_a_track_is_acquired():
    inner = _FakeDetector()
    td = TrackedDetector(inner, CAMS, detect_every=5)
    for _ in range(10):
        td.detect(FRAME, cam="cam_a")
    # Tick 1 forces a detect (no track yet); ticks 2-5 coast; tick 6 is due
    # again (since >= 5); ticks 7-10 coast. Two real network calls in ten ticks.
    assert td.calls == 2
    assert inner.calls == 2


def test_each_camera_gets_its_own_tracker_and_schedule():
    """A busy cam_a must not affect cam_b's schedule -- they are independent
    Kalman tracks, not a shared one keyed by call order."""
    inner = _FakeDetector()
    td = TrackedDetector(inner, CAMS, detect_every=5)
    for _ in range(3):
        td.detect(FRAME, cam="cam_a")
    for _ in range(1):
        td.detect(FRAME, cam="cam_b")
    # cam_a: tick 1 forced, ticks 2-3 coast -> 1 call. cam_b: tick 1 forced -> 1 call.
    assert td.calls == 2


def test_modes_are_tallied_across_cameras():
    inner = _FakeDetector()
    td = TrackedDetector(inner, CAMS, detect_every=5)
    for _ in range(5):
        td.detect(FRAME, cam="cam_a")
    assert td.modes["detect"] == 1
    assert td.modes["track"] == 4


def test_an_unrecognised_camera_falls_back_to_a_plain_detect():
    """A camera this instance was not built for still gets served, uncounted by
    any tracker -- but the call still counts, since the network still ran."""
    inner = _FakeDetector()
    td = TrackedDetector(inner, CAMS)
    result = td.detect(FRAME, cam="cam_never_configured")
    assert result == (10.0, 20.0)
    assert td.calls == 1
    assert "cam_never_configured" not in td._trackers


def test_detect_returns_none_when_the_tracker_has_nothing():
    inner = _NeverFindsDetector()
    td = TrackedDetector(inner, CAMS)
    assert td.detect(FRAME, cam="cam_a") is None


def test_a_real_trackedcamera_still_works_underneath():
    """Not just the adapter in isolation -- confirms it drives an actual
    TrackedCamera, not a reimplementation of its scheduling logic."""
    inner = _FakeDetector()
    td = TrackedDetector(inner, CAMS, detect_every=3)
    assert isinstance(td._trackers["cam_a"], TrackedCamera)
    x, y = td.detect(FRAME, cam="cam_a")
    assert (x, y) == (10.0, 20.0)


# --------------------------------------------------------------------------
# Cropping (`crop=True`): search a window around the prediction, not the
# whole frame, with a full-frame fallback when the window misses.
# --------------------------------------------------------------------------

FULL_FRAME = np.zeros((360, 640, 3), np.uint8)


class _ScriptedDetector:
    """Returns pre-programmed responses in call order. Records the shape of
    every image it was given, so a test can assert what was actually searched
    (the full frame or a smaller crop) without the detector needing any
    notion of what a crop is -- a plain ndarray slice carries no such thing."""

    name = "fake"
    target_offset_key = None

    def __init__(self, responses):
        self._responses = list(responses)
        self.shapes = []

    def detect(self, img, cam=None):
        self.shapes.append(img.shape[:2])
        return self._responses.pop(0)


def test_window_clips_to_the_frame_edges():
    t = TrackedCamera(crop_pad=30, cam="c")
    crop, (x0, y0) = t._window(np.zeros((100, 100, 3), np.uint8), 5, 5)
    assert (x0, y0) == (0, 0)                    # would be (-25, -25) unclipped
    assert crop.shape[:2] == (35, 35)


def test_no_crop_without_an_established_track():
    """Nothing to centre a window on yet -- the first detection is always a
    full-frame search, cropping or not."""
    det = _ScriptedDetector([(100.0, 100.0)])
    t = TrackedCamera(detector=det, detect_every=1, crop=True, crop_pad=20, cam="c")
    t.step(FULL_FRAME)
    assert det.shapes[0] == (360, 640)


def test_crop_centres_on_the_prediction_and_maps_the_offset_back():
    """Once a track exists, the search window follows the prediction, and a
    detection's coordinates -- local to that crop -- must be translated back
    to full-frame pixels before anything downstream sees them."""
    det = _ScriptedDetector([(300.0, 200.0),    # tick 1: acquire, full frame
                             (5.0, 5.0)])        # tick 2: found near the crop corner
    t = TrackedCamera(detector=det, detect_every=1, crop=True, crop_pad=20, cam="c")
    t.step(FULL_FRAME)
    x, y, mode = t.step(FULL_FRAME)

    assert mode == "detect"
    assert det.shapes[1] != (360, 640), "second call should have seen a crop"
    # Constant-velocity prediction from a track initialised with zero velocity
    # stays at (300, 200); a 20px window there has its top-left at (280, 180).
    assert (x, y) == pytest.approx((285.0, 185.0))


def test_crop_miss_falls_back_to_a_full_frame_search():
    det = _ScriptedDetector([(300.0, 200.0),    # tick 1: acquire
                             None,               # tick 2: crop search misses
                             (50.0, 60.0)])      # tick 2: full-frame retry finds it
    t = TrackedCamera(detector=det, detect_every=1, crop=True, crop_pad=20, cam="c")
    t.step(FULL_FRAME)
    x, y, mode = t.step(FULL_FRAME)

    assert mode == "detect"
    assert (x, y) == (50.0, 60.0)                # retry's own coords, no crop offset
    assert t.crop_misses == 1
    assert len(det.shapes) == 3                  # acquire + missed crop + full retry
    assert det.shapes[1] != (360, 640)
    assert det.shapes[2] == (360, 640)


def test_crop_false_is_unaffected_by_the_new_parameters():
    """Regression pin: the default path must be pixel-for-pixel what it was
    before cropping existed."""
    det = _ScriptedDetector([(300.0, 200.0), (301.0, 201.0)])
    t = TrackedCamera(detector=det, detect_every=1, cam="c")   # crop=False default
    t.step(FULL_FRAME)
    t.step(FULL_FRAME)
    assert det.shapes == [(360, 640), (360, 640)]
    assert t.crop_misses == 0
