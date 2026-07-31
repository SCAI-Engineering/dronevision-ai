"""Stale frames must be reported as absent, not served as current.

This exists because of a real failure. The camera service died mid-flight, and the
AI kept emitting a confident, perfectly frozen position for as long as it ran —
which the control software dutifully injected into the autopilot. Nothing errored.
Nothing warned. The estimate simply stopped changing, and a vehicle told it is
stationary at a point it has drifted away from will keep drifting.

Reporting nothing is strictly better: a consumer can detect absence, and both the
autopilot and the control software already handle it.

Tested against `NetSource`'s staleness logic directly rather than over a socket, so
this needs no publisher and no network.
"""
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq", reason="pyzmq not installed")

from dronevision.io.sources.net import NetSource  # noqa: E402

CAMS = ["cam_a", "cam_b"]


@pytest.fixture
def src():
    # Connecting is enough; nothing publishes on this endpoint, so no frames ever
    # arrive and the reader thread just times out harmlessly.
    s = NetSource(CAMS, endpoint="tcp://127.0.0.1:5999", max_age=0.2)
    yield s
    s.close()


def feed(src, cam, when=None, stamp=1.0):
    """Inject a frame as though it had arrived, bypassing the socket."""
    with src._lock:
        src._frames[cam] = np.zeros((8, 8, 3), np.uint8)
        src._meta[cam] = {"stamp": stamp, "seq": 1, "frame_id": cam}
        src._arrived[cam] = time.monotonic() if when is None else when


def test_a_camera_that_never_delivered_is_absent(src):
    assert src.latest("cam_a") is None
    assert src.meta("cam_a") is None
    assert src.stale() == CAMS
    assert src.age_s("cam_a") is None


def test_a_fresh_frame_is_served(src):
    feed(src, "cam_a")
    assert src.latest("cam_a") is not None
    assert src.meta("cam_a")["seq"] == 1
    assert "cam_a" not in src.stale()
    assert src.age_s("cam_a") < 0.2


def test_a_stale_frame_is_withheld(src):
    """The frame is still in memory — it must not be handed out anyway."""
    feed(src, "cam_a", when=time.monotonic() - 5.0)
    with src._lock:
        assert src._frames["cam_a"] is not None, "frame should still be buffered"
    assert src.latest("cam_a") is None
    assert src.meta("cam_a") is None
    assert "cam_a" in src.stale()


def test_staleness_is_per_camera(src):
    """One dead camera must not hide the others; the pipeline can still work with
    two of four, so withholding everything would be its own failure."""
    feed(src, "cam_a")
    feed(src, "cam_b", when=time.monotonic() - 5.0)
    assert src.latest("cam_a") is not None
    assert src.latest("cam_b") is None
    assert src.stale() == ["cam_b"]


def test_recovery_is_automatic(src):
    """The guard must not latch. A service that comes back must be used again
    without restarting the AI — verified live, and pinned here."""
    feed(src, "cam_a", when=time.monotonic() - 5.0)
    assert src.latest("cam_a") is None
    feed(src, "cam_a")                      # service returns
    assert src.latest("cam_a") is not None
    assert src.stale() == ["cam_b"]


def test_it_actually_expires_with_time(src):
    """Not just a flag — a frame that was fresh goes stale on its own."""
    feed(src, "cam_a")
    assert src.latest("cam_a") is not None
    time.sleep(0.25)                        # max_age is 0.2 s
    assert src.latest("cam_a") is None


def test_max_age_none_disables_the_check():
    """Only sensible for replay, where 'arrival time' has no meaning."""
    s = NetSource(CAMS, endpoint="tcp://127.0.0.1:5998", max_age=None)
    try:
        feed(s, "cam_a", when=time.monotonic() - 3600.0)
        assert s.latest("cam_a") is not None
        assert s.stale() == []
    finally:
        s.close()


def test_the_pipeline_produces_nothing_from_stale_frames(src):
    """End to end: withheld frames must mean no estimate, not a frozen one."""
    from dronevision.l2_perception.detector import ColorDetector
    from dronevision.l4_triangulation.calibration import load_site
    from dronevision.pipeline import LocalizationPipeline

    site = load_site("factory")
    stale_src = NetSource(site.cam_names, endpoint="tcp://127.0.0.1:5997",
                          max_age=0.2)
    try:
        for cam in site.cam_names:
            feed(stale_src, cam, when=time.monotonic() - 5.0)
        pipe = LocalizationPipeline(cal=site, detector=ColorDetector())
        assert pipe.locate_from(stale_src) is None
    finally:
        stale_src.close()


def test_replay_sources_have_no_staleness_concept():
    """A corpus is not a live feed; there is nothing to go stale, and a max_age
    check against wall clock would reject every frame on a slow machine."""
    from dronevision.io.sources.replay import ReplaySource

    assert not hasattr(ReplaySource, "max_age")
    assert not hasattr(ReplaySource, "stale")
