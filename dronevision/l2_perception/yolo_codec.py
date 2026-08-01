"""Pre- and post-processing for the detection network. numpy and OpenCV only.

Lives at layer level rather than inside `runtimes/` because this is a property of the
*network head*, shared by every runtime that consumes raw tensors, and unused by
Ultralytics (which does its own). Keeping it separate means adding a runtime touches one
file, and means the maths can be tested on a machine with no inference runtime installed.

THE OUTPUT LAYOUT IS SNIFFED, NEVER CONFIGURED. The bundled export is a YOLO26 one2one
head: it emits `(1, 300, 6)` of `x1,y1,x2,y2,conf,cls`, already sorted and already NMS-free.
A conventional v8-style decoder pointed at that tensor reads columns 4 and 5 as class
logits and produces confident, plausible, wrong boxes — no exception, just wrong
millimetres downstream. Quantized re-exports usually have to drop that head (static
quantization chokes on TopK/GatherElements), which brings back the raw `(1, 4+nc, A)`
layout. Both must work, and the code must decide which it is holding rather than trust a
flag someone forgot to set.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Layout(str, Enum):
    """How to read the network's output tensor."""

    END2END = "end2end"    # (1, N, 6)     x1,y1,x2,y2,conf,cls — sorted, no NMS needed
    RAW_CA = "raw_ca"      # (1, 4+nc, A)  channels-first: cx,cy,w,h then class scores
    RAW_AC = "raw_ac"      # (1, A, 4+nc)  the same, transposed


def anchor_count(imgsz, strides=(8, 16, 32)):
    """Predictions a v8/26-style head emits for a square input: sum of (imgsz/s)^2."""
    return sum((imgsz // s) ** 2 for s in strides)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Letterbox:
    """The resize that was applied, so boxes can be mapped back."""

    r: float          # scale factor applied to the source
    dw: float         # left padding, pixels
    dh: float         # top padding, pixels
    src_w: int
    src_h: int


def letterbox(img_rgb, size=320, pad=114, scaleup=True, out=None):
    """Resize preserving aspect ratio and pad to a square. Returns (canvas, Letterbox).

    `scaleup=True` matters more than it looks: the hybrid detector feeds crops far
    smaller than the network input, and refusing to enlarge them leaves a postage stamp in
    the corner of a grey canvas that the network cannot see. The default here differs from
    Ultralytics' inference default for exactly that reason.
    """
    h, w = img_rgb.shape[:2]
    r = min(size / w, size / h)
    if not scaleup:
        r = min(r, 1.0)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0

    if out is None or out.shape != (size, size, 3):
        out = np.empty((size, size, 3), np.uint8)
    out[:] = pad

    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    cv2.resize(img_rgb, (nw, nh), dst=out[top:top + nh, left:left + nw],
               interpolation=interp)
    return out, Letterbox(r=r, dw=left, dh=top, src_w=w, src_h=h)


def to_nchw(canvas_rgb, out=None):
    """HWC uint8 RGB -> NCHW float32 in [0,1], contiguous.

    No colour conversion. The network was trained on RGB; the BGR flip elsewhere in this
    package exists only because Ultralytics' ndarray path expects it and undoes it
    internally. Doing it here would swap red and blue and cost accuracy silently.
    """
    x = canvas_rgb.transpose(2, 0, 1)[None]
    if out is not None and out.shape == x.shape:
        np.multiply(x, np.float32(1.0 / 255.0), out=out, casting="unsafe")
        return out
    return np.ascontiguousarray(x, dtype=np.float32) * np.float32(1.0 / 255.0)


def unletterbox_xyxy(boxes, lb):
    """Map boxes from letterboxed canvas pixels back to source-frame pixels."""
    if len(boxes) == 0:
        return boxes
    out = boxes.astype(np.float32, copy=True)
    # Slices, not fancy indexing. `out[:, [0, 2]]` returns a COPY, so an in-place clip
    # with `out=` writes to a temporary and silently does nothing — the boxes then leave
    # the frame and the reprojection check downstream blames the detector.
    out[:, 0::2] = np.clip((out[:, 0::2] - lb.dw) / lb.r, 0, lb.src_w)
    out[:, 1::2] = np.clip((out[:, 1::2] - lb.dh) / lb.r, 0, lb.src_h)
    return out


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def sniff_layout(output_shapes, imgsz, nc=1, sample=None):
    """Work out how to read the output. Raises rather than guessing.

    `output_shapes` is the list of shapes from the runtime; `sample` is an actual output
    array from a warm-up pass, used to break the one ambiguous case.
    """
    if not output_shapes:
        raise ValueError("model has no outputs")
    shape = tuple(output_shapes[0])
    if len(shape) != 3:
        raise ValueError(
            f"expected a rank-3 detection output, got {shape}. This codec understands "
            f"(1,N,6) end-to-end and (1,4+nc,A) / (1,A,4+nc) raw heads.")

    _, a, b = shape
    expect_a = anchor_count(imgsz)
    ch = 4 + nc

    # THE AMBIGUOUS CASE FIRST. With two classes a raw head is also 6 columns wide, so
    # (1, A, 6) fits both readings and the shape rules below would confidently pick the
    # wrong one. Only the data distinguishes them, so check it before anything else.
    ambiguous = (b == 6 and ch == 6 and a == expect_a)
    if not ambiguous:
        if b == ch and a == expect_a:
            return Layout.RAW_AC
        if a == ch and b == expect_a:
            return Layout.RAW_CA
        if b == 6:
            return Layout.END2END

    # An end-to-end class column is integral and below nc, and its confidence column
    # lies in [0,1]. Raw class scores satisfy neither reliably.
    if b == 6 and sample is not None:
        d = np.asarray(sample).reshape(-1, 6)
        cls_integral = bool(np.all(np.equal(np.mod(d[:, 5], 1.0), 0.0)))
        conf_unit = bool(d[:, 4].min() >= -1e-6 and d[:, 4].max() <= 1.0 + 1e-6)
        if cls_integral and conf_unit:
            return Layout.END2END
        return Layout.RAW_AC

    raise ValueError(
        f"cannot identify the output layout: shape {shape}, imgsz {imgsz}, nc {nc}. "
        f"Expected (1,N,6) end-to-end, or a raw head with {ch} channels and "
        f"{expect_a} anchors.")


def check_box_scale(boxes, imgsz):
    """Reject a normalized export instead of silently reporting sub-pixel positions."""
    if len(boxes) == 0:
        return
    if float(np.abs(boxes[:, :4]).max()) <= 1.5:
        raise ValueError(
            "boxes appear to be normalized to [0,1]; this codec expects pixel "
            f"coordinates in the {imgsz}x{imgsz} input space. Re-export without "
            "normalization rather than scaling by assumption.")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def nms_xyxy(boxes, scores, iou=0.45, conf=0.25):
    """Indices surviving NMS, via OpenCV. Returns a flat int array (possibly empty).

    cv2.dnn.NMSBoxes takes xywh with a top-left origin, and its return type has changed
    across OpenCV versions — (N,1), (N,), or an empty tuple. Both are normalised here.
    """
    if len(boxes) == 0:
        return np.empty(0, np.int32)
    xywh = boxes[:, :4].copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(np.float32).tolist(),
                           float(conf), float(iou))
    return np.asarray(idx, dtype=np.int32).reshape(-1)


def decode(outputs, lb, layout, imgsz, nc=1, conf=0.25, iou=0.45, max_det=1, nms=None):
    """Raw output tensors -> (N,6) xyxy,conf,cls in SOURCE-frame pixels, best first.

    `nms=None` means "only when it can matter": with a single target the highest-scoring
    box is the answer and suppression is wasted work, so it is skipped when max_det == 1.
    """
    arr = np.asarray(outputs[0])

    if layout is Layout.END2END:
        d = arr.reshape(-1, 6)
        keep = d[:, 4] >= conf
        d = d[keep]
        # Already sorted by the head, but a re-export might not be.
        if len(d) > 1 and not np.all(np.diff(d[:, 4]) <= 1e-6):
            d = d[np.argsort(-d[:, 4])]
        check_box_scale(d, imgsz)
        d = d[:max_det] if max_det else d
        out = d.copy()
        out[:, :4] = unletterbox_xyxy(d[:, :4], lb)
        return out.astype(np.float32)

    p = arr[0]
    if layout is Layout.RAW_CA:
        p = p.T                                     # (A, 4+nc)

    if nc == 1:
        scores = p[:, 4]
        classes = np.zeros(len(p), np.float32)
    else:
        cls_block = p[:, 4:4 + nc]
        scores = cls_block.max(axis=1)
        classes = cls_block.argmax(axis=1).astype(np.float32)

    # Threshold before any box arithmetic: this drops thousands of anchors to a handful
    # and is most of the Python-side cost of decoding.
    keep = scores >= conf
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    p, scores, classes = p[keep], scores[keep], classes[keep]

    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    check_box_scale(boxes, imgsz)

    use_nms = (max_det is None or max_det > 1) if nms is None else nms
    if use_nms:
        sel = nms_xyxy(boxes, scores, iou=iou, conf=conf)
        if len(sel) == 0:
            return np.zeros((0, 6), np.float32)
        order = sel[np.argsort(-scores[sel])]
    else:
        order = np.argsort(-scores)
    if max_det:
        order = order[:max_det]

    out = np.empty((len(order), 6), np.float32)
    out[:, :4] = unletterbox_xyxy(boxes[order], lb)
    out[:, 4] = scores[order]
    out[:, 5] = classes[order]
    return out


def best_center(dets):
    """Centre of the highest-scoring detection as ``(u, v)``, or None."""
    if dets is None or len(dets) == 0:
        return None
    x1, y1, x2, y2 = dets[0, :4]
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
