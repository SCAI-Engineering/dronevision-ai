"""Worker-side entry points for the process backend of `parallel.py`.

Separate module, and everything at module scope, because the "spawn" start method
re-imports the worker's code in a fresh interpreter and pickles by qualified name — a
closure or a nested function cannot cross that boundary.

WHY SPAWN AND NOT FORK. Fork is the default on Linux and would be cheaper, but the child
inherits the parent's memory including any inference session and its thread pool. A
forked thread pool has no threads in the child, and the failure is a hang rather than an
error. Spawn costs a second of startup once, outside the measured region.
"""
import time

# One detector per process, built by the initializer. Module-global because that is the
# only state a spawned worker can carry between tasks.
_DETECTOR = None
_CAM = None


def init_worker(cam, detector, kwargs, warmup_shape=(360, 640, 3)):
    """Build this process's detector once, then warm it up.

    Warmup happens here rather than on the first real frame so that engine allocation
    and kernel selection stay out of the timed region — otherwise the first fix of every
    run carries a cost the rest do not.
    """
    global _DETECTOR, _CAM
    import numpy as np

    from dronevision.l2_perception.detector import make_detector

    _CAM = cam
    _DETECTOR = make_detector(detector, **(kwargs or {}))
    blank = np.zeros(warmup_shape, np.uint8)
    for _ in range(3):
        _DETECTOR.detect(blank, cam=cam)
    rt = getattr(_DETECTOR, "runtime", None)
    if rt is not None:
        from dronevision.l2_perception.runtimes.base import Stats
        rt.stats = Stats()


def detect_in_worker(frame):
    """Detect on one frame. Returns ``(uv, timings)``.

    Timings come back with the result because the parent cannot see into the child's
    memory, and the per-stage breakdown is exactly what distinguishes "the work
    overlapped" from "the work was serialized".
    """
    t0 = time.perf_counter()
    uv = _DETECTOR.detect(frame, cam=_CAM)
    wall = (time.perf_counter() - t0) * 1e3
    timings = dict(getattr(_DETECTOR, "last_timings", {}) or {})
    timings["worker_wall_ms"] = wall
    return uv, timings
