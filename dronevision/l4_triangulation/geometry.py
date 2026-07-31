"""Layer 4 — multi-view triangulation.

Two or more 2D detections of the same target become one 3D point, by DLT (direct
linear transform, solved with an SVD), followed by outlier rejection: the
candidate point is reprojected into every contributing camera, views that
disagree by more than a pixel threshold are discarded, and the point is re-solved
from what remains. That last part is what keeps the system alive when a camera is
occluded or a detector misfires.

Camera parameters come from a `Calibration` (see `calibration.py`) rather than
module constants. Two ways to use this:

    tri = Triangulator(load_site("factory"))     # preferred: explicit
    X, used = tri.triangulate(dets)

    X, used = triangulate(dets)                  # convenience: default site

The convenience form resolves the default site once, from the `DRONEVISION_SITE`
environment variable (default ``"factory"``). Prefer the explicit form in new
code — it makes the dependency visible and lets tests swap sites freely.
"""
import math
import os

import numpy as np

from dronevision.l4_triangulation.calibration import Calibration, load_site

__all__ = ["Triangulator", "dlt", "reproj_err", "triangulate",
           "default_site", "set_default_site"]

_DEFAULT = None


def default_site():
    """The process-wide default calibration, loaded once on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_site(os.environ.get("DRONEVISION_SITE", "factory"))
    return _DEFAULT


def set_default_site(cal):
    """Override the default calibration. Pass a `Calibration`, or None to reset."""
    global _DEFAULT
    if cal is not None and not isinstance(cal, Calibration):
        raise TypeError(f"expected a Calibration, got {type(cal).__name__}")
    _DEFAULT = cal


def _cal(cal):
    return default_site() if cal is None else cal


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------

def dlt(dets, cal=None):
    """Triangulate ``{cam: (u, v)}`` into a 3D world point. Needs >= 2 views.

    Each view contributes two linear constraints on the homogeneous point; the
    solution is the right null space of the stacked system, i.e. the singular
    vector of least singular value.
    """
    cal = _cal(cal)
    if len(dets) < 2:
        raise ValueError(f"triangulation needs at least 2 views, got {len(dets)}")
    rows = []
    for name, (u, v) in dets.items():
        P = cal[name].P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        # Degenerate: the views are parallel or coincident, so the point is at
        # infinity. Refusing beats returning a huge finite number.
        raise ValueError("degenerate view geometry; point is at infinity")
    return X[:3] / X[3]


def reproj_err(name, X, uv, cal=None):
    """Reprojection error in pixels of 3D point `X` against detection `uv`."""
    cal = _cal(cal)
    P = cal[name].P
    h = P @ np.append(np.asarray(X, float), 1.0)
    if abs(h[2]) < 1e-9:
        return math.inf
    return math.hypot(h[0] / h[2] - uv[0], h[1] / h[2] - uv[1])


def _consensus(dets, thresh, cal):
    """Find the largest set of views that agree, by minimal-subset consensus.

    Triangulate every pair, score each candidate point by how many views fall
    within `thresh` of it, and keep the best-supported set. Textbook RANSAC with
    an exhaustive rather than sampled search — affordable here because a pair is
    the minimal set for triangulation and four cameras give only six of them.

    Returns ``(inlier_names, total_error)``.
    """
    names = list(dets)
    best = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = {names[i]: dets[names[i]], names[j]: dets[names[j]]}
            try:
                X = dlt(pair, cal=cal)
            except ValueError:
                continue                      # degenerate pair; another will do
            errs = {n: reproj_err(n, X, uv, cal=cal) for n, uv in dets.items()}
            inliers = [n for n, e in errs.items() if e < thresh]
            # Most support wins; ties break toward the tighter fit.
            score = (len(inliers), -sum(errs[n] for n in inliers))
            if best is None or score > best[0]:
                best = (score, inliers)
    if best is None:
        return names, math.inf
    return best[1], -best[0][1]


def triangulate(dets, thresh_px=None, cal=None, method="consensus"):
    """Robust triangulation. Returns ``(X_world, [cameras_used])``.

    `thresh_px` defaults to the site's configured threshold. `method` selects how
    outliers are handled when there are more than two views:

    ``consensus``    (default) minimal-subset consensus — see `_consensus`.
    ``single_pass``  solve with every view, drop those whose reprojection error
                     exceeds the threshold, re-solve.

    WHY THE DEFAULT IS NOT ``single_pass``: that approach judges each view against
    a solution the outlier has already corrupted. One badly wrong detection pulls
    the point far enough that *most* views exceed the threshold — with four cameras
    and one detection off by ~134 px, three of the four measured 40-83 px of error,
    the liar among them, and the single view that looked acceptable was a good
    camera that happened to sit near the corrupted solution. Fewer than two views
    survive, so the "at least two must remain" guard declines to act at all, and
    the corrupted point is returned while still reporting every camera as used. It
    fails silently and reports full confidence, which is the worst available
    combination.

    Consensus costs six extra solves for four cameras — microseconds against tens
    of milliseconds for one detector inference, so robustness here is effectively
    free. `single_pass` is retained for comparison and reproducibility.

    With exactly two views neither method applies: there is nothing to
    cross-check against, so the result is accepted as-is.
    """
    cal = _cal(cal)
    thresh = cal.reproj_threshold_px if thresh_px is None else float(thresh_px)

    if len(dets) <= 2:
        return dlt(dets, cal=cal), sorted(dets)

    if method == "consensus":
        inliers, _ = _consensus(dets, thresh, cal)
        if len(inliers) >= 2:
            keep = {n: dets[n] for n in inliers}
            return dlt(keep, cal=cal), sorted(keep)
        return dlt(dets, cal=cal), sorted(dets)

    if method == "single_pass":
        X = dlt(dets, cal=cal)
        good = {n: uv for n, uv in dets.items()
                if reproj_err(n, X, uv, cal=cal) < thresh}
        if 2 <= len(good) < len(dets):
            X = dlt(good, cal=cal)
            dets = good
        return X, sorted(dets)

    raise ValueError(
        f"unknown method {method!r}; expected 'consensus' or 'single_pass'")


class Triangulator:
    """Triangulation bound to one site's calibration.

    Preferred over the module-level functions: the calibration is explicit, and
    several sites can coexist in one process (tests, multi-room deployments).
    """

    def __init__(self, cal=None, thresh_px=None):
        self.cal = _cal(cal)
        self.thresh_px = (self.cal.reproj_threshold_px if thresh_px is None
                          else float(thresh_px))

    @property
    def min_views(self):
        return self.cal.min_views

    def dlt(self, dets):
        return dlt(dets, cal=self.cal)

    def reproj_err(self, name, X, uv):
        return reproj_err(name, X, uv, cal=self.cal)

    def triangulate(self, dets):
        """``(X, used)``, or ``(None, [])`` if there are too few views.

        Returning None rather than raising is deliberate: dropping below the
        minimum view count is a routine event in a live pipeline (occlusion, a
        detector miss), not an error condition.
        """
        if len(dets) < self.cal.min_views:
            return None, []
        return triangulate(dets, thresh_px=self.thresh_px, cal=self.cal)


# ---------------------------------------------------------------------------
# Compatibility surface
# ---------------------------------------------------------------------------
# The pipeline previously read these as module constants. Resolved lazily through
# the default site so that importing this module performs no file I/O, and so a
# test can install its own site before anything touches them.

def __getattr__(name):
    if name == "K":
        cal = default_site()
        Ks = [c.K for c in cal.cameras.values()]
        if any(not np.allclose(k, Ks[0]) for k in Ks[1:]):
            raise AttributeError(
                "cameras have differing intrinsics; there is no single K. "
                "Use load_site(...)[cam].K instead.")
        return Ks[0]
    if name == "CAMS":
        return {n: (tuple(c.position), tuple(c.rpy))
                for n, c in default_site().cameras.items()}
    if name == "PROJ":
        return default_site().P
    if name == "MARKER_DZ":
        return default_site().marker_dz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
