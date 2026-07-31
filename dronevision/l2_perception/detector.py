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

from dronevision.l2_perception.paths import default_yolo, model_path
from dronevision.l2_perception.vision_utils import best_box_center, red_centroid


class Detector:
    """Common base. Subclasses implement `detect` and set `name`.

    `detects_marker` says *what* the reported pixel is centred on, which the
    caller needs in order to know whether to apply the marker's vertical offset:

        True   the marker mounted above the airframe -> subtract `marker_dz`
        False  the airframe itself -> no correction

    Getting this wrong does not fail visibly. It biases every reported altitude by
    the offset — 18 cm in the bundled site — which looks like a plausible
    calibration error rather than a bug.
    """

    name = "base"
    detects_marker = False

    def detect(self, img_rgb, cam=None):
        raise NotImplementedError


class ColorDetector(Detector):
    """Largest red blob in the frame — the marker, not the airframe."""

    name = "color"
    detects_marker = True

    def detect(self, img_rgb, cam=None):
        return red_centroid(img_rgb, "RGB")


class YoloDetector(Detector):
    """Ultralytics YOLO detecting the airframe by shape.

    This is the reference implementation, running via PyTorch. It is the baseline
    the optimized Arm backends are measured against — not itself the thing you
    would deploy to a Raspberry Pi.

    Trained against labels derived from the airframe's projected bounding box, so
    the reported centre is the airframe and no marker offset applies.
    """

    name = "yolo"
    detects_marker = False

    def __init__(self, model_path_=None, conf=0.25, imgsz=None):
        from ultralytics import YOLO           # imported late: heavy, and optional

        if model_path_ is None:
            model_path_ = os.environ.get("YOLO_MODEL") or default_yolo()
        if model_path_ is None:
            raise FileNotFoundError(
                "no detector weights found; put one in models/ or set YOLO_MODEL")
        self.path = model_path(model_path_)
        self.model = YOLO(str(self.path))
        self.conf = conf
        self.imgsz = imgsz
        # First inference allocates buffers and is far slower than the rest.
        # Doing it here keeps that cost out of the caller's timing.
        self.model.predict(np.zeros((360, 640, 3), np.uint8), verbose=False)

    def detect(self, img_rgb, cam=None):
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        kw = {"conf": self.conf, "verbose": False}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        r = self.model.predict(bgr, **kw)[0]
        c = best_box_center(r)
        return c[:2] if c else None


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
    detects_marker = False

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
    detects_marker = False

    def __init__(self, pad=28, coast=20, yolo=None):
        super().__init__(coast=coast)
        self.yolo = yolo if yolo is not None else YoloDetector()
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
