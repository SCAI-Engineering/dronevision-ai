"""Layer 2 — detectors. One image in, one pixel position out.

    detect(img_rgb, cam=None) -> (u, v), or None

Four backends behind that signature, chosen by `make_detector()` or the
`DETECTOR` environment variable:

    color    HSV threshold on a marker mounted on the airframe. Cheapest, and
             dependent on a marker existing — useful as a reference, not a
             deployment.
    yolo     a small network detecting the airframe by shape. The deployment
             path, and the only expensive one.
    motion   background subtraction (MOG2). Marker-free and scene-agnostic, but
             a stationary target disappears into the background, so it coasts on
             its last detection for a bounded number of frames.
    hybrid   motion proposes small regions, the network confirms only those.
             Cheaper than running the network over a full frame, and unlike bare
             motion it will not mistake a person for a drone.

`color` is the default because it needs no weights, so importing this module
never requires a model file to be present.

Note that *which network* and *which inference runtime executes it* are separate
choices. This module owns the first; runtime selection lives with the backends
under `runtimes/`, because the same weights perform very differently across Arm
cores depending on the instructions available.
"""
import os

import cv2
import numpy as np

from dronevision.l2_perception.paths import resolve_model
from dronevision.l2_perception.vision_utils import red_centroid
from dronevision.l2_perception.yolo_codec import best_center


class Detector:
    """Common base. Subclasses implement `detect` and set `name`.

    `target_offset_key` names WHAT the reported pixel is centred on, so the caller can
    look up the right vertical correction in the site config. Detectors do not all look
    at the same thing:

        "marker_dz"    a marker mounted well above the airframe (colour detector)
        "airframe_dz"  the airframe's bounding-box centroid (shape detectors)
        None           already the vehicle origin; no correction

    This was a boolean once, which was wrong: it could only express "marker or nothing",
    while the measured offsets differ per detector — 0.4333 m for the marker and 0.0950 m
    for a bounding-box centroid on the bundled site. Getting it wrong never fails
    visibly. It shifts every reported altitude by a constant and leaves the horizontal
    axes untouched, which reads as a calibration quirk rather than a bug.
    """

    name = "base"
    target_offset_key = None

    @property
    def detects_marker(self):
        """Kept for callers written against the older boolean."""
        return self.target_offset_key == "marker_dz"

    def detect(self, img_rgb, cam=None):
        raise NotImplementedError


class ColorDetector(Detector):
    """Largest red blob in the frame — the marker, not the airframe."""

    name = "color"
    target_offset_key = "marker_dz"

    def detect(self, img_rgb, cam=None):
        return red_centroid(img_rgb, "RGB")


class YoloDetector(Detector):
    """YOLO detecting the airframe by shape, executed by a swappable runtime.

    The network and the engine that runs it are independent choices: the same weights go
    through PyTorch, ONNX Runtime or ExecuTorch, selected with `runtime=`. Measuring the
    difference between those is what this project is for, so nothing here may assume one.

    `imgsz` DEFAULTS TO 320 AND IS NOT MERELY A HINT. Ultralytics left alone infers at 640
    with rectangular letterboxing, while an exported graph is a fixed 320x320 square —
    about 40% more pixels on one side. Comparing them would report a resolution artifact
    as a speedup. Every runtime is pinned to the same input size, and each reports the
    shape it actually ran so a results row can be checked.

    Trained against labels derived from the airframe's projected bounding box, so the
    reported centre is the airframe and no marker offset applies.
    """

    name = "yolo"
    target_offset_key = "airframe_dz"

    def __init__(self, model=None, conf=0.25, imgsz=320, runtime=None, threads=None,
                 iou=0.45, model_path_=None, **runtime_kw):
        from dronevision.l2_perception.runtimes import canonical, make_runtime

        model = model if model is not None else model_path_   # old keyword
        if model is None:
            model = os.environ.get("YOLO_MODEL")

        rt_name = canonical(runtime or os.environ.get("RUNTIME") or "ultralytics")
        self.path = resolve_model(model, rt_name)
        self.conf = conf
        self.iou = iou

        self.runtime = make_runtime(rt_name, model=self.path, imgsz=imgsz,
                                    threads=threads, **runtime_kw)
        # Instance attribute only: the class attribute stays "yolo" so BACKENDS lookups
        # and `detects_marker` logic keep working, while logs and results rows show which
        # engine actually ran.
        self.name = f"yolo/{self.runtime.name}"
        self.imgsz = self.runtime.imgsz
        self.runtime.warmup()

    def detect(self, img_rgb, cam=None):
        dets = self.runtime.detect_boxes(img_rgb, conf=self.conf, iou=self.iou,
                                         max_det=1)
        return best_center(dets)

    @property
    def last_timings(self):
        """Per-stage milliseconds for the most recent call: pre / infer / post."""
        return self.runtime.last_timings

    def describe(self):
        """Provenance for a results row — see `runtimes.base.InferenceRuntime`."""
        d = self.runtime.describe()
        d.update({"detector": "yolo", "conf": self.conf, "iou": self.iou})
        return d


class MotionDetector(Detector):
    """Background subtraction (MOG2), with per-camera background state.

    Fixed cameras make this viable: the background is genuinely static, so
    anything moving is interesting. The failure mode is a hovering target, which
    fades into the background model — hence `coast`, which holds the last
    detection for a bounded number of frames rather than reporting nothing.

    Area and aspect filters reject the obvious non-drones (a person is tall, a
    forklift is large).
    """

    name = "motion"
    target_offset_key = "airframe_dz"

    def __init__(self, coast=20, amin=8, amax=9000):
        self._mog = {}
        self._last = {}
        self.coast, self.amin, self.amax = coast, amin, amax

    def _fg_mask(self, img_rgb, cam):
        if cam not in self._mog:
            self._mog[cam] = cv2.createBackgroundSubtractorMOG2(
                history=150, varThreshold=40, detectShadows=True)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        fg = self._mog[cam].apply(bgr)
        # MOG2 marks shadows 127 and foreground 255; thresholding above 200 keeps
        # only true foreground, so the target's shadow is not tracked as a target.
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        return cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _coasted(self, cam):
        if cam in self._last and self._last[cam][2] < self.coast:
            self._last[cam][2] += 1
            return (self._last[cam][0], self._last[cam][1])
        return None

    def detect(self, img_rgb, cam="default"):
        fg = self._fg_mask(img_rgb, cam)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_a = None, 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.amin or a > self.amax:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not 0.4 <= w / max(h, 1) <= 2.5:        # the airframe is roughly square
                continue
            if a > best_a:
                best_a, best = a, (x + w / 2.0, y + h / 2.0)
        if best:
            self._last[cam] = [best[0], best[1], 0]
            return best
        return self._coasted(cam)


class HybridDetector(MotionDetector):
    """Motion proposes regions; the network confirms only those crops.

    Two wins over running the network on the whole frame: far fewer pixels
    through the network, and the network gets to veto a moving object that is not
    a drone — which bare motion detection cannot do.
    """

    name = "hybrid"
    target_offset_key = "airframe_dz"

    def __init__(self, pad=28, coast=20, yolo=None, **yolo_kw):
        super().__init__(coast=coast)
        # Runtime and thread settings must reach the inner detector, or a sweep would
        # silently benchmark the default engine while claiming to test another.
        self.yolo = yolo if yolo is not None else YoloDetector(**yolo_kw)
        self.pad = pad

    def detect(self, img_rgb, cam="default"):
        h_img, w_img = img_rgb.shape[:2]
        fg = self._fg_mask(img_rgb, cam)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Largest three regions only: past that the crops stop being cheaper than
        # one full-frame inference.
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(c) < 6:
                continue
            x, y, w, h = cv2.boundingRect(c)
            x0, y0 = max(0, x - self.pad), max(0, y - self.pad)
            x1, y1 = min(w_img, x + w + self.pad), min(h_img, y + h + self.pad)
            crop = img_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            px = self.yolo.detect(crop)
            if px:
                uv = (x0 + px[0], y0 + px[1])          # crop -> full-frame coords
                self._last[cam] = [uv[0], uv[1], 0]
                return uv
        return self._coasted(cam)


BACKENDS = {
    "color": ColorDetector,
    "yolo": YoloDetector,
    "motion": MotionDetector,
    "hybrid": HybridDetector,
}


def make_detector(name=None, **kw):
    """Build a detector by name, defaulting to `DETECTOR` then to ``"color"``."""
    name = (name or os.environ.get("DETECTOR", "color")).lower()
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown detector {name!r}; expected one of {sorted(BACKENDS)}") from None
    return cls(**kw)
