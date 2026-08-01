"""Pre/post-processing for the detection network.

These run with no inference runtime installed, which is the point: the codec is where a
silent wrongness would live, and it must be checkable on any machine.

The failure mode worth guarding is not a crash. A mis-read output layout, a letterbox
rounding error or a red/blue swap all produce plausible boxes at plausible coordinates,
and the only symptom is a few extra millimetres of position error indistinguishable from
quantization loss.
"""
import cv2
import numpy as np
import pytest

from dronevision.l2_perception.yolo_codec import (
    Layout, anchor_count, best_center, check_box_scale, decode, letterbox, nms_xyxy,
    sniff_layout, to_nchw, unletterbox_xyxy,
)


# --------------------------------------------------------------------------
# Letterbox round-trip — a box must come back where it started
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src_wh", [
    (640, 360),      # the camera frames
    (320, 320),      # already square
    (57, 43),        # a hybrid-detector crop: tiny and odd
    (1280, 720),
    (100, 400),      # taller than wide
])
def test_box_round_trips_through_letterbox(src_wh):
    w, h = src_wh
    img = np.zeros((h, w, 3), np.uint8)
    canvas, lb = letterbox(img, size=320)
    assert canvas.shape == (320, 320, 3)

    # Four corners-ish of the source, expressed as a box, mapped forward by hand and
    # back by the codec.
    src = np.array([[0.1 * w, 0.2 * h, 0.7 * w, 0.9 * h]], np.float32)
    fwd = src.copy()
    fwd[:, [0, 2]] = fwd[:, [0, 2]] * lb.r + lb.dw
    fwd[:, [1, 3]] = fwd[:, [1, 3]] * lb.r + lb.dh
    back = unletterbox_xyxy(fwd, lb)
    np.testing.assert_allclose(back, src, atol=1e-4)


def test_small_crops_are_enlarged_not_pasted_in_a_corner():
    """`scaleup` is why the hybrid detector works at all: a 57x43 crop left at native
    size would occupy 2% of the canvas and the network would see nothing."""
    img = np.zeros((43, 57, 3), np.uint8)
    _, up = letterbox(img, size=320, scaleup=True)
    _, no = letterbox(img, size=320, scaleup=False)
    assert up.r > 5.0, "a small crop must be enlarged to fill the input"
    assert no.r == 1.0


def test_letterbox_centres_the_image():
    _, lb = letterbox(np.zeros((360, 640, 3), np.uint8), size=320)
    assert lb.dw == 0                      # 640 is the long side: no horizontal padding
    assert lb.dh == pytest.approx(70, abs=1)   # (320 - 180) / 2


def test_unletterbox_clips_to_the_source_frame():
    _, lb = letterbox(np.zeros((360, 640, 3), np.uint8), size=320)
    out = unletterbox_xyxy(np.array([[-50.0, -50.0, 400.0, 400.0]], np.float32), lb)
    assert out[0, 0] >= 0 and out[0, 1] >= 0
    assert out[0, 2] <= 640 and out[0, 3] <= 360


def test_to_nchw_preserves_channel_order():
    """A red/blue swap here costs accuracy and raises nothing. The network was trained
    on RGB; the BGR flip elsewhere exists only for Ultralytics' ndarray convention."""
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :, 0] = 255                                   # pure red in RGB
    x = to_nchw(img)
    assert x.shape == (1, 3, 8, 8) and x.dtype == np.float32
    assert x[0, 0].mean() == pytest.approx(1.0)          # channel 0 hot
    assert x[0, 2].mean() == pytest.approx(0.0)          # channel 2 cold


# --------------------------------------------------------------------------
# Layout sniffing — the thing that must never guess
# --------------------------------------------------------------------------

def test_anchor_count():
    assert anchor_count(320) == 2100          # 40^2 + 20^2 + 10^2
    assert anchor_count(640) == 8400


def test_recognises_the_end2end_head():
    assert sniff_layout([(1, 300, 6)], imgsz=320, nc=1) is Layout.END2END


@pytest.mark.parametrize("shape,expect", [
    ((1, 5, 2100), Layout.RAW_CA),
    ((1, 2100, 5), Layout.RAW_AC),
])
def test_recognises_raw_heads(shape, expect):
    assert sniff_layout([shape], imgsz=320, nc=1) is expect


def test_ambiguous_six_column_shape_is_resolved_from_the_data():
    """(1, A, 6) with two classes collides with an end-to-end head. The data settles it:
    an end-to-end class column is integral and its confidence lies in [0,1]."""
    a = anchor_count(320)
    e2e = np.zeros((1, a, 6), np.float32)
    e2e[0, :, 4] = 0.5                     # confidence in range
    e2e[0, :, 5] = 1.0                     # integral class id
    assert sniff_layout([(1, a, 6)], imgsz=320, nc=2, sample=e2e) is Layout.END2END

    raw = np.zeros((1, a, 6), np.float32)
    raw[0, :, 4] = 3.7                     # class logits, not integral, not in [0,1]
    raw[0, :, 5] = 2.4
    assert sniff_layout([(1, a, 6)], imgsz=320, nc=2, sample=raw) is Layout.RAW_AC


def test_unrecognised_layout_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="cannot identify the output layout"):
        sniff_layout([(1, 17, 99)], imgsz=320, nc=1)
    with pytest.raises(ValueError, match="rank-3"):
        sniff_layout([(1, 300)], imgsz=320, nc=1)
    with pytest.raises(ValueError, match="no outputs"):
        sniff_layout([], imgsz=320, nc=1)


def test_normalized_boxes_are_rejected_not_rescaled():
    with pytest.raises(ValueError, match="normalized"):
        check_box_scale(np.array([[0.1, 0.2, 0.3, 0.4, 0.9, 0.0]], np.float32), 320)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def _lb(w=640, h=360):
    return letterbox(np.zeros((h, w, 3), np.uint8), size=320)[1]


def test_end2end_decode_places_the_box_correctly():
    lb = _lb()
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [150, 120, 170, 140, 0.9, 0.0]      # canvas pixels
    out[0, 1] = [10, 10, 20, 20, 0.01, 0.0]         # below threshold

    d = decode([out], lb, Layout.END2END, 320, conf=0.25, max_det=1)
    assert d.shape == (1, 6)
    # canvas -> source: /0.5 horizontally, and 70 px of top padding removed
    assert d[0, 0] == pytest.approx(300.0)
    assert d[0, 1] == pytest.approx(100.0)
    assert d[0, 4] == pytest.approx(0.9)


def test_end2end_decode_drops_everything_below_threshold():
    lb = _lb()
    out = np.zeros((1, 300, 6), np.float32)
    out[0, :, 4] = 0.05
    assert len(decode([out], lb, Layout.END2END, 320, conf=0.25)) == 0


def test_raw_decode_converts_centre_form_to_corners():
    lb = _lb()
    a = anchor_count(320)
    out = np.zeros((1, 5, a), np.float32)
    out[0, :, 7] = [160, 130, 20, 10, 0.8]           # cx, cy, w, h, score

    d = decode([out], lb, Layout.RAW_CA, 320, conf=0.25, max_det=1)
    assert d.shape == (1, 6)
    # centre (160,130) size 20x10 -> canvas corners (150,125)-(170,135)
    assert d[0, 0] == pytest.approx(300.0)           # (150 - 0) / 0.5
    assert d[0, 1] == pytest.approx(110.0)           # (125 - 70) / 0.5
    assert d[0, 4] == pytest.approx(0.8)


def test_both_layouts_agree_on_the_same_target():
    """The two heads describe the same box differently; decoding must reconcile them."""
    lb = _lb()
    e2e = np.zeros((1, 300, 6), np.float32)
    e2e[0, 0] = [150, 125, 170, 135, 0.8, 0.0]
    a = anchor_count(320)
    raw = np.zeros((1, 5, a), np.float32)
    raw[0, :, 3] = [160, 130, 20, 10, 0.8]

    da = decode([e2e], lb, Layout.END2END, 320, max_det=1)
    db = decode([raw], lb, Layout.RAW_CA, 320, max_det=1)
    np.testing.assert_allclose(da[:, :4], db[:, :4], atol=1e-4)


def test_multi_class_raw_uses_the_best_class():
    lb = _lb()
    a = anchor_count(320)
    out = np.zeros((1, 6, a), np.float32)            # nc = 2
    out[0, :, 5] = [160, 130, 20, 10, 0.2, 0.7]      # class 1 wins
    d = decode([out], lb, Layout.RAW_CA, 320, nc=2, conf=0.25, max_det=1)
    assert len(d) == 1
    assert d[0, 5] == pytest.approx(1.0)
    assert d[0, 4] == pytest.approx(0.7)


def test_single_target_skips_nms_but_still_returns_the_best():
    """With max_det=1 the top score is the answer, so suppression is wasted work."""
    lb = _lb()
    a = anchor_count(320)
    out = np.zeros((1, 5, a), np.float32)
    out[0, :, 0] = [100, 100, 20, 20, 0.4]
    out[0, :, 1] = [104, 102, 20, 20, 0.9]           # overlapping, stronger
    d = decode([out], lb, Layout.RAW_CA, 320, conf=0.25, max_det=1)
    assert len(d) == 1 and d[0, 4] == pytest.approx(0.9)


def test_nms_collapses_overlapping_boxes_when_asked():
    boxes = np.array([[100, 100, 140, 140], [104, 102, 144, 142],
                      [300, 300, 340, 340]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms_xyxy(boxes, scores, iou=0.45, conf=0.25)
    assert len(keep) == 2                     # the near-duplicate is suppressed
    assert 0 in keep and 2 in keep


def test_nms_on_empty_input_returns_an_empty_array():
    out = nms_xyxy(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    assert len(out) == 0


def test_best_center():
    assert best_center(None) is None
    assert best_center(np.zeros((0, 6), np.float32)) is None
    d = np.array([[10.0, 20.0, 30.0, 40.0, 0.9, 0.0]], np.float32)
    assert best_center(d) == (20.0, 30.0)


# --------------------------------------------------------------------------
# A synthetic end-to-end pass: the codec must find a box it was handed
# --------------------------------------------------------------------------

def test_a_known_pixel_survives_the_whole_round_trip():
    """Draw a target at a known place, pretend the network found it in canvas space,
    and check the decoded position lands back on the original pixel."""
    img = np.zeros((360, 640, 3), np.uint8)
    truth = (412.0, 233.0)
    cv2.circle(img, (int(truth[0]), int(truth[1])), 6, (255, 0, 0), -1)

    canvas, lb = letterbox(img, size=320)
    cu, cv_ = truth[0] * lb.r + lb.dw, truth[1] * lb.r + lb.dh
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [cu - 7, cv_ - 7, cu + 7, cv_ + 7, 0.95, 0.0]

    got = best_center(decode([out], lb, Layout.END2END, 320, max_det=1))
    assert got[0] == pytest.approx(truth[0], abs=0.5)
    assert got[1] == pytest.approx(truth[1], abs=0.5)
