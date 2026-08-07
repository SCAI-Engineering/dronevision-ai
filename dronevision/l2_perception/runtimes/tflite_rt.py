"""TensorFlow Lite / LiteRT backend for CPU and Ethos-U execution."""
import os
import time
from pathlib import Path

import numpy as np

from dronevision.l2_perception import yolo_codec as codec
from dronevision.l2_perception.runtimes.base import InferenceRuntime, TensorSpec, Threads


def _get_interpreter(model_path, threads=1, delegate_path=None):
    """Instantiate a LiteRT / TFLite interpreter with optional delegate."""
    interpreter_cls = None
    errors = []

    for mod_name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            mod = __import__(mod_name, fromlist=["Interpreter", "load_delegate"])
            interpreter_cls = getattr(mod, "Interpreter")
            load_delegate_fn = getattr(mod, "load_delegate", None)
            break
        except ImportError as exc:
            errors.append(f"{mod_name}: {exc}")

    if interpreter_cls is None:
        raise RuntimeError(
            "No LiteRT / TFLite interpreter module found.\n"
            f"Errors: {'; '.join(errors)}"
        )

    delegates = []
    if delegate_path:
        if not load_delegate_fn:
            raise RuntimeError(f"load_delegate is not available in {mod_name}")
        delegates.append(load_delegate_fn(delegate_path))

    kw = {}
    if hasattr(interpreter_cls, "__init__"):
        import inspect
        sig = inspect.signature(interpreter_cls.__init__)
        if "num_threads" in sig.parameters:
            kw["num_threads"] = threads

    if delegates:
        kw["experimental_delegates"] = delegates

    return interpreter_cls(model_path=str(model_path), **kw)


class TfLiteRuntime(InferenceRuntime):
    """Executes a `.tflite` / `.litert` graph on CPU or Ethos-U delegate."""

    name = "tflite"

    def __init__(self, model, imgsz=None, threads=None, nc=1, delegate_path=None, **kw):
        super().__init__(model, imgsz=imgsz, threads=threads, **kw)
        self.nc = nc
        self.delegate_path = delegate_path
        self._canvas = None
        self._layout = None

        self.interpreter = _get_interpreter(
            self.model_path,
            threads=self.threads.intra,
            delegate_path=self.delegate_path,
        )
        self.interpreter.allocate_tensors()

        self._in_details = self.interpreter.get_input_details()[0]
        self._out_details = self.interpreter.get_output_details()[0]

        self.in_shape = tuple(self._in_details["shape"])
        self.in_dtype = np.dtype(self._in_details["dtype"])
        self.in_quant = self._in_details.get("quantization", (0.0, 0))

        self.out_shape = tuple(self._out_details["shape"])
        self.out_dtype = np.dtype(self._out_details["dtype"])
        self.out_quant = self._out_details.get("quantization", (0.0, 0))

        # Determine spatial size from graph input
        if len(self.in_shape) == 4:
            if self.in_shape[1] == 3:  # NCHW
                self.imgsz = int(self.in_shape[2])
            else:  # NHWC
                self.imgsz = int(self.in_shape[1])
        elif self.imgsz:
            self.imgsz = int(self.imgsz)

    def _prepare_input(self, canvas_rgb):
        scale, zero_point = self.in_quant
        h, w, c = canvas_rgb.shape

        if len(self.in_shape) == 4 and self.in_shape[1] == 3:  # NCHW
            x = canvas_rgb.transpose(2, 0, 1)[None]
        else:  # NHWC
            x = canvas_rgb[None]

        if self.in_dtype == np.float32:
            return x.astype(np.float32) / 255.0

        if scale > 0.0:
            # Scale float [0, 1] to integer tensor using scale and zero_point
            x_float = x.astype(np.float32) / 255.0
            q_min = float(np.iinfo(self.in_dtype).min)
            q_max = float(np.iinfo(self.in_dtype).max)
            x_quant = np.round(x_float / scale + zero_point).clip(q_min, q_max)
            return x_quant.astype(self.in_dtype)

        return x.astype(self.in_dtype)

    def _dequantize_output(self, raw_out):
        scale, zero_point = self.out_quant
        if self.out_dtype != np.float32 and scale > 0.0:
            return (raw_out.astype(np.float32) - zero_point) * scale
        return raw_out.astype(np.float32)

    def detect_boxes(self, img_rgb, conf=0.25, iou=0.45, max_det=1):
        size = self.imgsz or 320
        self.imgsz = size

        t0 = time.perf_counter()
        canvas, lb = codec.letterbox(img_rgb, size=size, scaleup=True, out=self._canvas)
        self._canvas = canvas
        input_data = self._prepare_input(canvas)
        t1 = time.perf_counter()

        self.interpreter.set_tensor(self._in_details["index"], input_data)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self._out_details["index"])
        t2 = time.perf_counter()

        outs = [self._dequantize_output(raw_out)]

        # LiteRT/TFLite exporters normalize box coordinates (cx, cy, w, h) to [0, 1].
        # Scale box channels back to pixel space [0, size] if normalized (< size / 2.0).
        o = outs[0]
        if o.ndim == 3:
            if o.shape[1] == 4 + self.nc and o.shape[1] >= 4:  # (1, 4+nc, A)
                if float(np.abs(o[:, :4, :]).max()) < (size / 2.0):
                    o[:, :4, :] *= float(size)
            elif o.shape[2] == 4 + self.nc and o.shape[2] >= 4:  # (1, A, 4+nc)
                if float(np.abs(o[:, :, :4]).max()) < (size / 2.0):
                    o[:, :, :4] *= float(size)

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

    @property
    def input_spec(self):
        return TensorSpec(self._in_details["name"], self.in_shape, str(self.in_dtype))

    @property
    def output_specs(self):
        return (TensorSpec(self._out_details["name"], self.out_shape, str(self.out_dtype)),)

    def describe(self):
        d = super().describe()
        d.update({
            "input": f"{self._in_details['name']}{list(self.in_shape)}",
            "input_dtype": str(self.in_dtype),
            "input_quant": self.in_quant,
            "outputs": [f"{self._out_details['name']}{list(self.out_shape)}"],
            "output_dtype": str(self.out_dtype),
            "output_quant": self.out_quant,
            "layout": self._layout.value if self._layout else None,
            "delegate": self.delegate_path or "none",
            "precision": "int8" if self.in_dtype in (np.int8, np.uint8) else "fp32",
        })
        return d
