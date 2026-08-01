"""What a runtime is, and what it must report about itself.

A *detector* decides what to look for; a *runtime* decides what executes the network.
They are separate axes: the same weights run under PyTorch, ONNX Runtime and (later)
ExecuTorch, and the whole project exists to measure the difference between them.

THE SEAM IS `detect_boxes`, NOT `infer`. Ultralytics owns its own pre- and
post-processing; raw-tensor runtimes do not. Putting the contract at the box level lets
both fit without contorting either, and `TensorRuntime` implements it once for everything
that speaks NCHW.

EVERY RUNTIME MUST DESCRIBE ITSELF. `describe()` returns the provenance for one row of the
results table — model hash, input shape, precision, thread counts, providers. A latency
number without that context is not comparable to any other number, and the benchmark
harness refuses to record one.
"""
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dronevision.l2_perception import yolo_codec as codec


@dataclass(frozen=True)
class Threads:
    """Thread counts, pinned so that two measurements mean the same thing.

    These settings are PROCESS-GLOBAL in every backend that honours them (OpenCV and
    torch's interop pool cannot be changed once work has started). Two runtimes with
    different values in one process silently share whichever was applied last, so the
    sweep runs one configuration per process.
    """

    intra: int = 1
    inter: int = 1
    opencv: int = None          # None -> follow `intra`

    @classmethod
    def resolve(cls, spec=None):
        """From an explicit value, else DV_THREADS, else all cores."""
        if isinstance(spec, Threads):
            return spec
        if spec is None:
            spec = os.environ.get("DV_THREADS")
        if spec is None:
            return cls(intra=os.cpu_count() or 1)
        return cls(intra=int(spec))

    @property
    def cv(self):
        return self.intra if self.opencv is None else self.opencv

    def apply_opencv(self):
        import cv2
        cv2.setNumThreads(self.cv)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    dtype: str


@dataclass
class Stats:
    """Running per-stage totals, so a benchmark can report means without re-timing."""

    n: int = 0
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    def add(self, pre, inf, post):
        self.n += 1
        self.pre_ms += pre
        self.infer_ms += inf
        self.post_ms += post

    def means(self):
        n = max(self.n, 1)
        return {"pre_ms": self.pre_ms / n, "infer_ms": self.infer_ms / n,
                "post_ms": self.post_ms / n,
                "total_ms": (self.pre_ms + self.infer_ms + self.post_ms) / n}


def file_digest(path, n=12):
    """Short content hash, so a results row names the exact artifact measured."""
    try:
        h = hashlib.sha256()
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    h.update(f.name.encode())
                    h.update(f.read_bytes())
        else:
            h.update(p.read_bytes())
        return h.hexdigest()[:n]
    except OSError:
        return "unknown"


class InferenceRuntime:
    """One way of executing the detection network."""

    name = "base"

    def __init__(self, model, imgsz=None, threads=None, **kw):
        self.model_path = str(model)
        self.imgsz = imgsz
        self.threads = Threads.resolve(threads)
        self.last_timings = {}
        self.stats = Stats()

    def detect_boxes(self, img_rgb, conf=0.25, iou=0.45, max_det=1):
        """``(N, 6)`` of ``x1,y1,x2,y2,conf,cls`` in SOURCE-frame pixels, best first."""
        raise NotImplementedError

    def warmup(self, n=3, size=None):
        """Run a few throwaway frames. The first inference allocates buffers and picks
        kernels; leaving that in the measurement inflates the first row of every sweep."""
        size = size or (360, 640)
        blank = np.zeros((size[0], size[1], 3), np.uint8)
        for _ in range(n):
            self.detect_boxes(blank)
        self.stats = Stats()

    def describe(self):
        """Provenance for one results row. Subclasses extend, never replace."""
        return {
            "runtime": self.name,
            "model": Path(self.model_path).name,
            "model_sha12": file_digest(self.model_path),
            "imgsz": self.imgsz,
            "threads_intra": self.threads.intra,
            "threads_inter": self.threads.inter,
            "threads_opencv": self.threads.cv,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TensorRuntime(InferenceRuntime):
    """A runtime that takes NCHW float32 and returns raw tensors.

    Implements `detect_boxes` once, so a new backend only supplies `infer()` and its
    tensor specs. The output layout is sniffed on the first real call and then reused.
    """

    def __init__(self, model, imgsz=None, threads=None, nc=1, **kw):
        super().__init__(model, imgsz=imgsz, threads=threads, **kw)
        self.nc = nc
        self._layout = None
        self._canvas = None
        self.threads.apply_opencv()

    # -- to be provided by the backend --------------------------------------

    def infer(self, x_nchw):
        raise NotImplementedError

    @property
    def input_spec(self):
        raise NotImplementedError

    @property
    def output_specs(self):
        raise NotImplementedError

    # -- shared implementation ----------------------------------------------

    @property
    def layout(self):
        return self._layout

    def _resolve_imgsz(self):
        """Prefer the graph's own input size over anything the caller guessed."""
        shape = self.input_spec.shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])
        if self.imgsz:
            return int(self.imgsz)
        raise ValueError(
            f"{self.name}: model has a dynamic input {shape} and no imgsz was given")

    def detect_boxes(self, img_rgb, conf=0.25, iou=0.45, max_det=1):
        size = self.imgsz or self._resolve_imgsz()
        self.imgsz = size

        t0 = time.perf_counter()
        canvas, lb = codec.letterbox(img_rgb, size=size, scaleup=True, out=self._canvas)
        self._canvas = canvas
        x = codec.to_nchw(canvas)
        t1 = time.perf_counter()

        outs = self.infer(x)
        t2 = time.perf_counter()

        if self._layout is None:
            self._layout = codec.sniff_layout(
                [o.shape for o in outs], size, nc=self.nc, sample=outs[0])
        dets = codec.decode(outs, lb, self._layout, size, nc=self.nc,
                            conf=conf, iou=iou, max_det=max_det)
        t3 = time.perf_counter()

        pre, inf, post = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        self.last_timings = {"pre_ms": pre, "infer_ms": inf, "post_ms": post}
        self.stats.add(pre, inf, post)
        return dets

    def describe(self):
        d = super().describe()
        try:
            d.update({
                "input": f"{self.input_spec.name}{list(self.input_spec.shape)}",
                "input_dtype": self.input_spec.dtype,
                "outputs": [f"{o.name}{list(o.shape)}" for o in self.output_specs],
                "layout": self._layout.value if self._layout else None,
            })
        except NotImplementedError:
            pass
        return d
