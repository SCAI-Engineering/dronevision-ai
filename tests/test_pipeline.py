"""End-to-end pipeline tests against synthetic imagery.

Renders a red marker into four camera views at the correct projected pixel for a
known 3D position, runs the real pipeline over them, and checks the position that
comes back. No Gazebo, no drone, no network — so this runs anywhere, including on
the Raspberry Pi it is meant to be deployed to.

Synthetic imagery cannot tell you whether the *detector* is any good; the recorded
corpus does that. What it does test is every other link in the chain — projection,
association, triangulation, outlier rejection, the marker offset, smoothing — with
a ground truth that is exact by construction.
"""
import cv2
import numpy as np
import pytest

from dronevision.l3_association.associator import DetectionAssociator
from dronevision.l3_association.tracker import TrackedCamera
from dronevision.l5_estimation.smoothing import EMASmoother
from dronevision.l2_perception.detector import (
    ColorDetector, Detector, make_detector,
)
from dronevision.pipeline import Estimate, LocalizationPipeline
from dronevision.io.sources.base import FrameSource
from dronevision.l4_triangulation.calibration import load_site
from dronevision.l4_triangulation.geometry import Triangulator

SITE = load_site("factory")


def render(cam, xyz, radius=7, noise=0):
    """A camera view containing a red blob where `xyz` actually projects."""
    w, h = cam.resolution
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 45, 50)                       # dull background, nothing near red
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(np.int16)
                      + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    uv = cam.project(xyz)
    if uv is not None:
        cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), radius,
                   (255, 0, 0), -1)             # RGB order: pure red
    return img


def frames_for(xyz, cams=None, **kw):
    cams = cams or SITE.cam_names
    return {c: render(SITE[c], xyz, **kw) for c in cams}


class StubSource(FrameSource):
    """A `FrameSource` over a fixed dict of frames, for testing consumers."""

    def __init__(self, frames, stamp=100.0):
        self.cams = list(frames)
        self._frames = frames
        self._stamp = stamp

    def latest(self, cam):
        return self._frames.get(cam)

    def meta(self, cam):
        return {"stamp": self._stamp, "seq": 1, "frame_id": cam}


# --------------------------------------------------------------------------
# The whole chain
# --------------------------------------------------------------------------

# Tolerances are set by pixel quantisation of the rendered blob, not by the
# geometry — which is exact to 1e-9 given exact pixels (see test_calibration).
# One pixel of detection error maps to roughly 2 cm in the interior with four
# cameras, and roughly 3.5 cm at the room edge where only two cameras see the
# target and their baseline is shallow. Those figures are the useful output here.
@pytest.mark.parametrize("truth, n_cams, tol_m", [
    ((0.0, 0.0, 2.5), 4, 0.05),
    ((3.5, -2.0, 1.8), 4, 0.05),
    ((-6.0, 5.5, 3.6), 4, 0.05),
    ((8.0, 0.0, 2.5), 2, 0.10),      # room edge: only two cameras cover this
])
def test_locates_a_known_position(truth, n_cams, tol_m):
    """The marker sits `marker_dz` above the airframe, so that is what comes back."""
    truth = np.array(truth, float)
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate(frames_for(truth))

    assert est is not None
    expected = truth - np.array([0.0, 0.0, SITE.marker_dz])
    err = np.linalg.norm(np.array(est.position) - expected)
    assert err < tol_m, f"{err * 100:.1f} cm from truth at {truth}"
    assert len(est.cams_used) == n_cams


def test_marker_offset_is_applied_only_when_the_detector_sees_the_marker():
    """A shape detector centres on the airframe; subtracting the offset then puts
    the drone 18 cm below where it is. This is the regression that guards it."""
    truth = np.array([1.0, 1.0, 2.5])

    class ShapeDetector(Detector):
        name = "shape"
        detects_marker = False

        def detect(self, img_rgb, cam=None):
            from dronevision.l2_perception.vision_utils import red_centroid
            return red_centroid(img_rgb, "RGB")

    marker = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                  smoother=EMASmoother(0.0))
    shape = LocalizationPipeline(cal=SITE, detector=ShapeDetector(),
                                 smoother=EMASmoother(0.0))
    f = frames_for(truth)

    assert marker.marker_correction is True
    assert shape.marker_correction is False
    dz = shape.locate(f).position[2] - marker.locate(f).position[2]
    assert dz == pytest.approx(SITE.marker_dz, abs=1e-9)


def test_explicit_marker_correction_overrides_the_detector():
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                marker_correction=False)
    assert pipe.marker_correction is False


# --------------------------------------------------------------------------
# Degraded conditions
# --------------------------------------------------------------------------

def test_two_cameras_are_enough():
    """Occlusion tolerance: the loop must survive losing half the cameras."""
    truth = np.array([2.0, -1.0, 2.2])
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0),
                                occlude=["cam_ne", "cam_sw"])
    est = pipe.locate(frames_for(truth))
    assert est is not None
    assert sorted(est.cams_used) == ["cam_nw", "cam_se"]
    expected = truth - np.array([0.0, 0.0, SITE.marker_dz])
    assert np.linalg.norm(np.array(est.position) - expected) < 0.15


def test_one_camera_yields_nothing():
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                occlude=["cam_ne", "cam_nw", "cam_sw"])
    assert pipe.locate(frames_for([0.0, 0.0, 2.5])) is None


def test_empty_frames_yield_nothing():
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector())
    blank = {c: np.zeros((360, 640, 3), np.uint8) for c in SITE.cam_names}
    assert pipe.locate(blank) is None


def test_missing_frames_are_tolerated():
    """A camera that has not delivered yet must not break the others."""
    truth = np.array([0.5, 0.5, 2.0])
    f = frames_for(truth)
    f["cam_ne"] = None
    est = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                              smoother=EMASmoother(0.0)).locate(f)
    assert est is not None and len(est.cams_used) == 3


def test_a_lying_camera_is_rejected():
    """One camera reporting a wrong pixel must be discarded, not averaged in.

    The bad blob is placed at a deliberately offset *pixel*, not by rendering a
    different 3D point — a wrong 3D point may simply project outside the image,
    in which case the camera reports nothing and outlier rejection is never
    exercised at all.
    """
    truth = np.array([1.0, 2.0, 2.5])
    f = frames_for(truth)
    cam = SITE["cam_se"]
    u, v = cam.project(truth)
    liar = np.zeros((cam.resolution[1], cam.resolution[0], 3), np.uint8)
    liar[:] = (40, 45, 50)
    cv2.circle(liar, (int(u) + 120, int(v) + 60), 7, (255, 0, 0), -1)
    f["cam_se"] = liar

    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate(f)
    assert est is not None
    assert "cam_se" in [c for c in SITE.cam_names] and "cam_se" not in est.cams_used
    assert est.n_rejected == 1
    expected = truth - np.array([0.0, 0.0, SITE.marker_dz])
    assert np.linalg.norm(np.array(est.position) - expected) < 0.10


def test_survives_image_noise():
    truth = np.array([-2.0, 3.0, 2.0])
    est = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                              smoother=EMASmoother(0.0)).locate(
        frames_for(truth, noise=25))
    assert est is not None
    expected = truth - np.array([0.0, 0.0, SITE.marker_dz])
    assert np.linalg.norm(np.array(est.position) - expected) < 0.10


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def test_locate_from_a_frame_source():
    truth = np.array([1.5, -2.0, 2.4])
    src = StubSource(frames_for(truth), stamp=1234.5)
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate_from(src)
    assert est is not None
    assert est.src_stamp == pytest.approx(1234.5)


def test_estimate_converts_to_the_wire_format():
    truth = np.array([1.0, 1.0, 2.0])
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(), timing=True)
    est = pipe.locate(frames_for(truth), src_stamp=500.0)
    state = est.to_state(seq=7)

    assert state.seq == 7
    assert state.src_t == 500.0
    assert state.quality == pytest.approx(1.0)
    assert len(state.to_json()) < 512          # must fit one datagram
    assert set(est.timings_ms) == {"detect", "associate", "triangulate", "smooth"}


def test_smoothing_lags_a_step_change():
    """The EMA is a low-pass filter, so it must not track a jump instantly. That
    lag is real and lands in the control loop; this pins the behaviour."""
    a, b = np.array([0.0, 0.0, 2.0]), np.array([4.0, 0.0, 2.0])
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.5))
    pipe.locate(frames_for(a))
    est = pipe.locate(frames_for(b))
    assert 1.5 < est.position[0] < 2.5          # roughly halfway, not at 4.0


def test_reset_clears_history():
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.5))
    pipe.locate(frames_for([0.0, 0.0, 2.0]))
    pipe.reset()
    est = pipe.locate(frames_for([4.0, 0.0, 2.0]))
    assert est.position[0] == pytest.approx(4.0, abs=0.05)


def test_sequence_counter_advances_only_on_success():
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector())
    pipe.locate(frames_for([0.0, 0.0, 2.0]))
    assert pipe.seq == 1
    pipe.locate({c: np.zeros((360, 640, 3), np.uint8) for c in SITE.cam_names})
    assert pipe.seq == 1


# --------------------------------------------------------------------------
# Detector registry and per-camera tracking
# --------------------------------------------------------------------------

def test_make_detector_default_and_registry():
    assert make_detector().name == "color"
    assert make_detector("color").detects_marker is True
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("nonexistent")


def test_tracked_camera_detects_then_coasts():
    """The whole throughput argument rests on this: the detector runs on a
    fraction of frames, and the estimate stays usable in between."""
    cam = SITE["cam_ne"]
    tracker = TrackedCamera(detector=ColorDetector(), detect_every=5, cam="cam_ne")

    modes = []
    for i in range(20):
        xyz = np.array([0.0 + i * 0.05, 0.0, 2.5])       # steady drift
        x, y, mode = tracker.step(render(cam, xyz))
        modes.append(mode)
        assert x is not None, f"lost the target at frame {i} (mode={mode})"

    assert modes[0] == "detect"
    assert modes.count("detect") <= 5                    # ran on <= 1 frame in 4
    assert modes.count("track") >= 14
    assert tracker.detect_rate <= 0.3


def test_tracked_camera_coasting_stays_close_to_truth():
    """Coasting is only worth having if the coasted position is usable."""
    cam = SITE["cam_ne"]
    tracker = TrackedCamera(detector=ColorDetector(), detect_every=4, cam="cam_ne")
    worst = 0.0
    for i in range(24):
        xyz = np.array([-2.0 + i * 0.08, 1.0, 2.5])
        x, y, _ = tracker.step(render(cam, xyz))
        u, v = cam.project(xyz)
        worst = max(worst, float(np.hypot(x - u, y - v)))
    assert worst < 12.0, f"coasted estimate drifted {worst:.1f} px from truth"


def test_tracked_camera_gives_up_on_a_vanished_target():
    cam = SITE["cam_ne"]
    tracker = TrackedCamera(detector=ColorDetector(), detect_every=3,
                            max_miss=5, cam="cam_ne")
    tracker.step(render(cam, np.array([0.0, 0.0, 2.5])))
    blank = np.zeros((cam.resolution[1], cam.resolution[0], 3), np.uint8)
    modes = [tracker.step(blank)[2] for _ in range(15)]
    assert modes[-1] == "idle"
    assert not tracker.have_track


# --------------------------------------------------------------------------
# Triangulator surface
# --------------------------------------------------------------------------

def test_triangulator_returns_none_below_min_views():
    tri = Triangulator(SITE)
    assert tri.triangulate({"cam_ne": (100.0, 100.0)}) == (None, [])


def test_dlt_rejects_a_single_view():
    with pytest.raises(ValueError, match="at least 2 views"):
        Triangulator(SITE).dlt({"cam_ne": (100.0, 100.0)})


def test_associator_drops_empty_detections():
    a = DetectionAssociator()
    assert a.associate({}) == {}
    assert a.associate({"cam_ne": None}) == {}
    assert a.associate({"cam_ne": (1.0, 2.0), "cam_nw": None}) == {0: {"cam_ne": (1.0, 2.0)}}


def test_estimate_reports_rejections():
    e = Estimate(position=(0, 0, 0), raw=(0, 0, 0),
                 cams_used=["a", "b"], n_detections=4)
    assert e.n_rejected == 2


# --------------------------------------------------------------------------
# Which instant an estimate describes (`src_stamp` / `newest_stamp` / `mean_stamp`)
# --------------------------------------------------------------------------

class SkewedSource(StubSource):
    """A `StubSource` whose cameras carry *different* capture stamps.

    `StubSource` gives every camera the same stamp, which is exactly the case
    that cannot distinguish the three -- so it cannot catch an estimate indexed
    by the wrong end of the capture window.
    """

    def __init__(self, frames, stamps):
        super().__init__(frames)
        self._stamps = stamps

    def meta(self, cam):
        return {"stamp": self._stamps[cam], "seq": 1, "frame_id": cam}


def test_mean_stamp_is_the_average_of_the_capture_window():
    """Triangulation weights the cameras roughly equally, so its output
    describes the mean capture instant -- not `src_stamp` (a staleness bound)
    and not `newest_stamp`. Indexing ground truth by either endpoint charges
    half the cross-camera skew to the estimator as if it were error."""
    truth = np.array([1.0, -1.0, 2.4])
    stamps = {"cam_ne": 10.00, "cam_nw": 10.04, "cam_sw": 10.08, "cam_se": 10.12}
    src = SkewedSource(frames_for(truth), stamps)
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate_from(src)
    assert est is not None
    assert est.src_stamp == pytest.approx(10.00)
    assert est.newest_stamp == pytest.approx(10.12)
    assert est.mean_stamp == pytest.approx(10.06)


def test_all_three_stamps_coincide_without_skew():
    """A caller with only one time to give must not end up with a mean of
    zero: with no skew information the three are the same instant."""
    truth = np.array([0.5, 0.5, 2.5])
    src = StubSource(frames_for(truth), stamp=7.5)
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate_from(src)
    assert est is not None
    assert est.src_stamp == pytest.approx(7.5)
    assert est.newest_stamp == pytest.approx(7.5)
    assert est.mean_stamp == pytest.approx(7.5)


def test_locate_defaults_mean_stamp_to_src_stamp():
    """`locate()` is the lower-level entry point; a caller that passes only
    `src_stamp` gets it echoed rather than a silent zero."""
    truth = np.array([0.0, 0.0, 2.5])
    pipe = LocalizationPipeline(cal=SITE, detector=ColorDetector(),
                                smoother=EMASmoother(0.0))
    est = pipe.locate(frames_for(truth), src_stamp=42.0)
    assert est is not None
    assert est.mean_stamp == pytest.approx(42.0)
    assert est.newest_stamp == pytest.approx(42.0)
