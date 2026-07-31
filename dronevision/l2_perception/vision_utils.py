"""Detector helpers that need only numpy and OpenCV — no neural network.

Colour-marker detection, and pulling the best box out of an inference result.
Kept separate from `detector.py` so the cheap classical path carries no
dependency on any inference runtime.
"""
import cv2
import numpy as np

# HSV bounds for the red marker. Red wraps around the hue origin, so it takes two
# ranges — a single one would miss half the marker.
_RED_LO1, _RED_HI1 = (0, 100, 60), (12, 255, 255)
_RED_LO2, _RED_HI2 = (168, 100, 60), (180, 255, 255)


def red_centroid(img, order="RGB", min_area=3):
    """Centroid ``(u, v)`` of the largest red blob, or None.

    `order` must match the channel order of `img`: frames arriving from the
    camera transport are RGB, frames already through a cv2 conversion are usually
    BGR. Getting it wrong does not raise — it silently thresholds the blue
    channel as though it were red — hence the explicit argument.
    """
    code = cv2.COLOR_RGB2HSV if order.upper() == "RGB" else cv2.COLOR_BGR2HSV
    hsv = cv2.cvtColor(img, code)
    mask = cv2.inRange(hsv, _RED_LO1, _RED_HI1) | cv2.inRange(hsv, _RED_LO2, _RED_HI2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def best_box(result):
    """``(x1, y1, x2, y2, conf)`` of the highest-confidence box, or None."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    i = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    return (x1, y1, x2, y2, float(boxes.conf[i]))


def best_box_center(result):
    """Centre ``(cx, cy, conf)`` of the highest-confidence box, or None."""
    b = best_box(result)
    if b is None:
        return None
    x1, y1, x2, y2, conf = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf)
