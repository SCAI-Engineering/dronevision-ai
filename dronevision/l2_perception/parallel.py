"""Run the detector on several cameras at once, one per core.

WHY THIS IS THE FLAGSHIP OPTIMIZATION. Inference on this network is memory-bound, so
handing one image to four threads returns only 2.19x on four Arm cores — measured on a
Cortex-A72, against a 4x ideal. The parallelism is in the wrong place. Four cameras are
four independent images, which is embarrassingly parallel and should scale far closer to
linearly: give each core a whole frame and a single-threaded engine instead of splitting
one frame four ways.

    sequential, 4 threads each   4 x 93.7 ms = 375 ms per fix   2.7 Hz
    parallel,   1 thread each      ~205 ms   per fix            4.9 Hz

Same silicon, same total work, different decomposition.

THREADS OR PROCESSES. The reflex is multiprocessing, because of the GIL. But ONNX Runtime
releases the GIL for the duration of a `Run` call, and so do OpenCV's resize and JPEG
decode, so threads can genuinely overlap here and cost nothing to feed — a process worker
would need every frame pickled across a pipe. Threads are the default for that reason;
`backend="process"` exists so the assumption can be tested rather than trusted, and
`bench/parallel.py` measures both.

EACH WORKER OWNS ITS OWN DETECTOR. Not only for thread-safety: the motion and hybrid
backends keep a per-camera background model, so a shared instance would be both a race
and a correctness bug. One detector per camera makes the state naturally private.
"""
import os
from concurrent.futures import ThreadPoolExecutor

DEFAULT_BACKEND = "thread"


def _worker_threads(n_workers, total=None):
    """How many threads each worker's engine should use.

    Oversubscription is the classic way this optimization gets slower instead of faster:
    four workers each spinning up four intra-op threads puts sixteen runnable threads on
    four cores, and they spend their time descheduling each other.
    """
    total = total or os.cpu_count() or 1
    return max(1, total // max(1, n_workers))


class ParallelPerception:
    """Detect across cameras concurrently, one detector per camera.

    `detect_all({cam: image})` returns `{cam: (u, v)}` for the cameras that found
    something, and is a drop-in replacement for looping over cameras.
    """

    def __init__(self, cams, factory, workers=None, backend=DEFAULT_BACKEND,
                 threads_per_worker=None, total_threads=None):
        """`factory(cam)` builds the detector for one camera.

        It is a callable rather than an instance because each worker needs its own engine
        and its own state, and because the number of threads that engine should use
        depends on how many workers there are — which the caller does not know until here.
        """
        self.cams = list(cams)
        self.backend = backend
        self.workers = int(workers or len(self.cams))
        self.threads_per_worker = (
            threads_per_worker if threads_per_worker is not None
            else _worker_threads(self.workers, total_threads))

        if backend == "process":
            raise NotImplementedError(
                "the process backend is not implemented: ONNX Runtime releases the GIL "
                "during inference, so threads already overlap and cost nothing to feed, "
                "whereas processes would pickle every frame across a pipe. Measure with "
                "bench/parallel.py before reaching for it.")
        if backend != "thread":
            raise ValueError(f"unknown backend {backend!r}; expected 'thread'")

        # Built eagerly and in series: each one loads weights and runs a warmup, and
        # doing that concurrently would contend for exactly the cores being measured.
        self.detectors = {c: factory(c, self.threads_per_worker) for c in self.cams}
        self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix="detect")

    @property
    def name(self):
        first = next(iter(self.detectors.values()), None)
        return f"parallel[{self.workers}x{self.threads_per_worker}t]/" + (
            getattr(first, "name", "?"))

    @property
    def target_offset_key(self):
        """Inherited from the wrapped detectors; they are all the same kind."""
        first = next(iter(self.detectors.values()), None)
        return getattr(first, "target_offset_key", None)

    def detect(self, img_rgb, cam=None):
        """Single-camera path, so this can stand in for a plain detector."""
        det = self.detectors.get(cam) or next(iter(self.detectors.values()))
        return det.detect(img_rgb, cam=cam)

    def detect_all(self, frames):
        """``{cam: rgb}`` -> ``{cam: (u, v)}``, computed concurrently."""
        work = [(c, img) for c, img in frames.items()
                if img is not None and c in self.detectors]
        if not work:
            return {}
        if len(work) == 1:
            # One camera: the pool would only add a hand-off.
            cam, img = work[0]
            uv = self.detectors[cam].detect(img, cam=cam)
            return {cam: uv} if uv else {}

        futures = {cam: self._pool.submit(self.detectors[cam].detect, img, cam=cam)
                   for cam, img in work}
        out = {}
        for cam, fut in futures.items():
            uv = fut.result()
            if uv:
                out[cam] = uv
        return out

    def worker_stats(self):
        """Per-worker mean stage times, and what they imply about the bottleneck.

        This is the diagnostic that separates the two reasons camera-parallelism can
        underperform, which have opposite remedies:

          * each worker's own inference stays fast but the wall time does not fall —
            the work is not overlapping, so something is serializing it (the GIL), and
            processes would help.
          * each worker's inference slows down by roughly the worker count — the work IS
            overlapping but contending for a shared resource, almost always memory
            bandwidth on a single-channel SoC. Processes would change nothing.
        """
        out = {}
        for cam, d in self.detectors.items():
            rt = getattr(d, "runtime", None)
            if rt is not None and getattr(rt, "stats", None) and rt.stats.n:
                out[cam] = {"n": rt.stats.n, **{k: round(v, 3)
                                                for k, v in rt.stats.means().items()}}
        return out

    def reset_stats(self):
        for d in self.detectors.values():
            rt = getattr(d, "runtime", None)
            if rt is not None:
                from dronevision.l2_perception.runtimes.base import Stats
                rt.stats = Stats()

    def describe(self):
        first = next(iter(self.detectors.values()), None)
        d = dict(first.describe()) if hasattr(first, "describe") else {}
        d.update({
            "parallel_backend": self.backend,
            "parallel_workers": self.workers,
            "threads_per_worker": self.threads_per_worker,
            "effective_threads": self.workers * self.threads_per_worker,
        })
        return d

    def close(self):
        self._pool.shutdown(wait=True)
        for d in self.detectors.values():
            rt = getattr(d, "runtime", None)
            if rt is not None:
                rt.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_parallel(cams, detector="yolo", workers=None, backend=DEFAULT_BACKEND,
                  total_threads=None, **detector_kw):
    """Convenience: one detector per camera, threads divided among the workers."""
    from dronevision.l2_perception.detector import make_detector

    def factory(cam, threads):
        kw = dict(detector_kw)
        if detector in ("yolo", "hybrid"):
            kw["threads"] = threads
        return make_detector(detector, **kw)

    return ParallelPerception(cams, factory, workers=workers, backend=backend,
                              total_threads=total_threads)
