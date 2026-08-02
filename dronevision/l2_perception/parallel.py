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

# Safe at module scope: `_procworker` imports only the standard library until a worker
# actually starts, so this does not pull an inference runtime into the parent.
from dronevision.l2_perception._procworker import (
    detect_in_worker as _detect_in_worker,
    init_worker as _init_worker,
)

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

        if backend not in ("thread", "process"):
            raise ValueError(
                f"unknown backend {backend!r}; expected 'thread' or 'process'")

        self._procs = None
        self._proc_stats = {}
        self.detectors = {}
        if backend == "process":
            self._start_processes(factory)
            self._pool = None
        else:
            # Built eagerly and in series: each one loads weights and runs a warmup, and
            # doing that concurrently would contend for exactly the cores being measured.
            self.detectors = {c: factory(c, self.threads_per_worker) for c in self.cams}
            self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix="detect")

    def _start_processes(self, factory):
        """One single-worker pool per camera.

        Not one pool of N workers: a pool hands a task to whichever worker is free, so a
        camera's frames would land in different processes run to run. That breaks the
        motion and hybrid detectors, whose background model is per-camera state, and it
        costs cache locality even for the stateless ones. A dedicated worker per camera
        reproduces exactly the affinity the thread backend gets for free.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        spec = getattr(factory, "worker_spec", None)
        if spec is None:
            raise ValueError(
                "the process backend needs a picklable description of the detector, "
                "not a factory closure: a spawned worker re-imports the code in a fresh "
                "interpreter and cannot receive a local function. Use make_parallel(), "
                "which attaches `worker_spec`.")
        detector_name, kwargs = spec
        kwargs = dict(kwargs or {})
        kwargs["threads"] = self.threads_per_worker

        ctx = mp.get_context("spawn")
        self._procs = {}
        for cam in self.cams:
            self._procs[cam] = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx,
                initializer=_init_worker, initargs=(cam, detector_name, kwargs))
        # Force each interpreter to start and warm up now, so process creation is not
        # charged to the first measured frame.
        import numpy as np
        blank = np.zeros((8, 8, 3), np.uint8)
        for cam, pool in self._procs.items():
            pool.submit(_detect_in_worker, blank).result()

    @property
    def name(self):
        first = next(iter(self.detectors.values()), None)
        tag = "proc" if self._procs is not None else "thread"
        base = getattr(first, "name", None) or getattr(self, "_detector_name", "?")
        return f"parallel-{tag}[{self.workers}x{self.threads_per_worker}t]/{base}"

    @property
    def target_offset_key(self):
        """Inherited from the wrapped detectors; they are all the same kind.

        With process workers the detector lives in another interpreter, so this comes
        from the class rather than an instance — getting it wrong would bias every
        altitude, so it is looked up rather than defaulted.
        """
        first = next(iter(self.detectors.values()), None)
        if first is not None:
            return getattr(first, "target_offset_key", None)
        from dronevision.l2_perception.detector import BACKENDS
        cls = BACKENDS.get(getattr(self, "_detector_name", None))
        return getattr(cls, "target_offset_key", None) if cls else None

    def detect(self, img_rgb, cam=None):
        """Single-camera path, so this can stand in for a plain detector."""
        det = self.detectors.get(cam) or next(iter(self.detectors.values()))
        return det.detect(img_rgb, cam=cam)

    def detect_all(self, frames):
        """``{cam: rgb}`` -> ``{cam: (u, v)}``, computed concurrently."""
        if self._procs is not None:
            return self._detect_all_processes(frames)

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

    def _detect_all_processes(self, frames):
        """Same contract, but each frame is pickled across a pipe to its worker.

        That transfer is a genuine cost of this backend and is deliberately included in
        the timing: it is what you pay to escape the GIL, and the comparison is only
        honest if it is charged.
        """
        futures = {}
        for cam, img in frames.items():
            if img is not None and cam in self._procs:
                futures[cam] = self._procs[cam].submit(_detect_in_worker, img)
        out = {}
        for cam, fut in futures.items():
            uv, timings = fut.result()
            self._proc_stats.setdefault(cam, []).append(timings)
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
        if self._procs is not None:
            out = {}
            for cam, rows in self._proc_stats.items():
                if not rows:
                    continue
                keys = {k for r in rows for k in r}
                out[cam] = {"n": len(rows)}
                out[cam].update({k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 3)
                                 for k in sorted(keys)})
            return out

        out = {}
        for cam, d in self.detectors.items():
            rt = getattr(d, "runtime", None)
            if rt is not None and getattr(rt, "stats", None) and rt.stats.n:
                out[cam] = {"n": rt.stats.n, **{k: round(v, 3)
                                                for k, v in rt.stats.means().items()}}
        return out

    def reset_stats(self):
        self._proc_stats = {}
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
        if self._procs is not None:
            for pool in self._procs.values():
                pool.shutdown(wait=True)
            self._procs = None
            return
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
                  total_threads=None, threads_per_worker=None, **detector_kw):
    """Convenience: one detector per camera, threads divided among the workers.

    `threads_per_worker` must be passed HERE, not set on the returned object: the engines
    are constructed during __init__, so assigning it afterwards relabels the
    configuration without changing what ran.
    """
    from dronevision.l2_perception.detector import make_detector

    def factory(cam, threads):
        kw = dict(detector_kw)
        if detector in ("yolo", "hybrid"):
            kw["threads"] = threads
        return make_detector(detector, **kw)

    # The process backend cannot receive a closure: a spawned worker re-imports the
    # code in a fresh interpreter. Carry a picklable description alongside it.
    factory.worker_spec = (detector, dict(detector_kw))

    par = ParallelPerception(cams, factory, workers=workers, backend=backend,
                             total_threads=total_threads,
                             threads_per_worker=threads_per_worker)
    par._detector_name = detector
    return par
