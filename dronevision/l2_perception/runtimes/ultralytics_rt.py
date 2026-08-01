"""Ultralytics / PyTorch backend — the reference every other runtime is measured against.

Not a deployment target. Torch on a Raspberry Pi is large and slow; this exists so that
"is the ONNX path correct?" and "how much did we gain?" have an answer produced by the
code the model was trained and validated with.

IT MUST BE FORCED TO THE SAME INPUT SIZE AS EVERY OTHER RUNTIME. Left to itself,
Ultralytics infers at 640 with rectangular letterboxing, so a 640x360 frame becomes about
320x192 — roughly 40% fewer pixels than the square 320x320 an exported graph consumes.
Benchmarking those two against each other measures different amounts of work and yields a
speedup figure that is a resolution artifact rather than a result. `imgsz` and
`rect=False` are therefore not optional here.
"""
import numpy as np

from dronevision.l2_perception.runtimes.base import InferenceRuntime


class UltralyticsRuntime(InferenceRuntime):
    """Runs a `.pt` through Ultralytics, which does its own pre- and post-processing."""

    name = "ultralytics"

    def __init__(self, model, imgsz=320, threads=None, **kw):
        super().__init__(model, imgsz=int(imgsz), threads=threads, **kw)
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                f"Ultralytics/PyTorch are not installed ({e}).\n"
                f"  pip install -e \".[torch]\"") from None

        torch.set_num_threads(self.threads.intra)
        try:
            torch.set_num_interop_threads(self.threads.inter)
        except RuntimeError:
            # Only settable once per process, before any parallel work. A second runtime
            # in the same process inherits the first one's setting; describe() reports
            # what was actually applied rather than what was asked for.
            pass
        self.threads.apply_opencv()
        self._torch = torch
        self.model = YOLO(str(self.model_path))

    def detect_boxes(self, img_rgb, conf=0.25, iou=0.45, max_det=1):
        import time

        import cv2

        t0 = time.perf_counter()
        # Ultralytics' ndarray path expects BGR and flips it back internally. Runtimes
        # fed raw tensors must NOT do this; see yolo_codec.to_nchw.
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        t1 = time.perf_counter()

        r = self.model.predict(bgr, imgsz=self.imgsz, rect=False, conf=conf, iou=iou,
                               max_det=max_det, device="cpu", verbose=False)[0]
        t2 = time.perf_counter()

        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            dets = np.zeros((0, 6), np.float32)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            cf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cl = boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, cf, cl]).astype(np.float32)
            dets = dets[np.argsort(-dets[:, 4])][:max_det]
        t3 = time.perf_counter()

        pre, inf, post = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        self.last_timings = {"pre_ms": pre, "infer_ms": inf, "post_ms": post}
        self.stats.add(pre, inf, post)
        return dets

    def describe(self):
        d = super().describe()
        import torch
        import ultralytics
        d.update({
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "torch_threads_applied": torch.get_num_threads(),
            "providers": ["torch-cpu"],
            "precision": "fp32",
            "input": f"images[1, 3, {self.imgsz}, {self.imgsz}]",
            "input_dtype": "float32",
            "note": "reference path; rect=False forces square letterboxing for parity",
        })
        return d
