"""Outlier rejection in triangulation.

The pipeline's occlusion and misdetection tolerance is entirely this code. It is
tested separately from the pixel-level pipeline tests because the interesting
inputs are exact detections plus a deliberate lie, with no rendering in between to
blur the numbers.

`test_single_pass_fails_where_consensus_succeeds` documents the specific failure
that motivated the default: it is a characterisation test, and it is expected to
keep failing for `single_pass` forever.
"""
import numpy as np
import pytest

from dronevision.l4_triangulation.calibration import load_site
from dronevision.l4_triangulation.geometry import dlt, reproj_err, triangulate

SITE = load_site("factory")
TRUTH = np.array([1.0, 2.0, 2.5])


def exact(truth=TRUTH, cams=None):
    """Exact projected pixels — no quantisation, no rendering."""
    return {c: SITE[c].project(truth) for c in (cams or SITE.cam_names)}


def with_liar(cam="cam_se", du=120.0, dv=60.0, truth=TRUTH):
    d = exact(truth)
    u, v = d[cam]
    d[cam] = (u + du, v + dv)
    return d


# --------------------------------------------------------------------------

def test_clean_data_is_exact():
    X, used = triangulate(exact(), cal=SITE)
    assert np.linalg.norm(X - TRUTH) < 1e-9
    assert used == sorted(SITE.cam_names)


def test_consensus_and_single_pass_agree_when_there_is_no_outlier():
    """The robust default must not perturb the normal case."""
    a, ua = triangulate(exact(), cal=SITE, method="consensus")
    b, ub = triangulate(exact(), cal=SITE, method="single_pass")
    np.testing.assert_allclose(a, b, atol=1e-9)
    assert ua == ub


@pytest.mark.parametrize("cam", ["cam_ne", "cam_nw", "cam_sw", "cam_se"])
def test_a_gross_outlier_is_identified_and_dropped(cam):
    X, used = triangulate(with_liar(cam), cal=SITE)
    assert cam not in used, f"{cam} lied by ~134 px and was kept"
    assert len(used) == 3
    assert np.linalg.norm(X - TRUTH) < 1e-9, "surviving views should be exact"


def test_single_pass_fails_where_consensus_succeeds():
    """Characterisation of the inherited algorithm's breakdown.

    Judging views against a solution the outlier already corrupted inverts the
    verdict: the good cameras look wrong and the liar looks right. Fewer than two
    views survive, so the rejection step declines to act and returns the corrupted
    point — while reporting every camera as used.
    """
    dets = with_liar()

    bad, bad_used = triangulate(dets, cal=SITE, method="single_pass")
    good, good_used = triangulate(dets, cal=SITE, method="consensus")

    # The old path is badly wrong and claims full confidence.
    assert np.linalg.norm(bad - TRUTH) > 2.0
    assert len(bad_used) == 4

    # The default is exact and names the culprit by omission.
    assert np.linalg.norm(good - TRUTH) < 1e-9
    assert good_used == ["cam_ne", "cam_nw", "cam_sw"]


def test_the_corrupted_solution_indicts_almost_everything():
    """The mechanism itself, so the reason is recorded and not just the symptom.

    A single lie drags the least-squares point ~2.4 m off truth. Judged against
    *that*, three of four views exceed the threshold — the liar included. Only one
    view looks acceptable, and it is an honest camera that happens to sit near the
    corrupted solution. So the guard requiring two survivors is never satisfied
    and nothing is dropped: the failure is not "the liar looks innocent" but
    "there is no one left to trust".
    """
    dets = with_liar()
    X = dlt(dets, cal=SITE)                       # contaminated
    errs = {c: reproj_err(c, X, dets[c], cal=SITE) for c in dets}
    thresh = SITE.reproj_threshold_px

    assert np.linalg.norm(X - TRUTH) > 2.0
    over = [c for c, e in errs.items() if e >= thresh]
    assert len(over) == 3, errs
    assert "cam_se" in over                       # the liar is flagged too
    survivors = [c for c in errs if c not in over]
    assert len(survivors) == 1                    # below the minimum of two
    assert survivors[0] != "cam_se"               # and it is an honest camera


def test_two_gross_outliers_still_leave_a_usable_answer():
    """Two of four lying is the limit — the two honest cameras remain the
    largest agreeing set only because the liars disagree with each other too."""
    dets = with_liar("cam_se")
    u, v = dets["cam_ne"]
    dets["cam_ne"] = (u - 90.0, v + 110.0)

    X, used = triangulate(dets, cal=SITE)
    assert sorted(used) == ["cam_nw", "cam_sw"]
    assert np.linalg.norm(X - TRUTH) < 1e-8


def test_small_errors_are_kept_not_rejected():
    """Sub-threshold noise is signal, not outliers: rejecting it would throw away
    views and make the estimate worse rather than better.

    Two pixels of per-camera noise costs roughly 11 cm of 3D error here, which is
    the number that sets how accurate the detector actually has to be.
    """
    rng = np.random.default_rng(3)
    dets = {c: (u + rng.normal(0, 2.0), v + rng.normal(0, 2.0))
            for c, (u, v) in exact().items()}
    X, used = triangulate(dets, cal=SITE)
    assert len(used) == 4
    assert np.linalg.norm(X - TRUTH) < 0.15


def test_two_views_are_accepted_without_cross_checking():
    """With two views an outlier is indistinguishable from the truth, so the
    result is returned as-is rather than pretending to have validated it."""
    dets = exact(cams=["cam_nw", "cam_sw"])
    X, used = triangulate(dets, cal=SITE)
    assert used == ["cam_nw", "cam_sw"]
    assert np.linalg.norm(X - TRUTH) < 1e-9


def test_threshold_is_taken_from_the_site_config():
    dets = with_liar(du=30.0, dv=0.0)
    # Loose enough to call the 30 px offset agreement; tight enough to reject it.
    assert len(triangulate(dets, thresh_px=200.0, cal=SITE)[1]) == 4
    assert "cam_se" not in triangulate(dets, thresh_px=5.0, cal=SITE)[1]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        triangulate(exact(), cal=SITE, method="magic")
