"""Calibration tests.

The load-bearing one is `test_matches_hardcoded_simulator_geometry`: moving camera
parameters out of code and into configuration is only safe if the projection
matrices come out bit-for-bit unchanged. A silent discrepancy here would not
crash anything — it would quietly bias every 3D position the system reports.
"""
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

from dronevision.l4_triangulation.calibration import (
    AXIS_CONVENTIONS, Calibration, intrinsic_matrix, load_site, rpy_to_R,
)

SITE = load_site("factory")

# The reference implementation lives in the companion simulator project, which is
# not present in a fresh clone of this repository. Point SIMULATOR_PATH at it to
# enable the equivalence test.
SIM_GEOMETRY = Path(
    os.environ.get("SIMULATOR_PATH", Path(__file__).resolve().parents[2] / "Simulator")
) / "tools" / "ai" / "triangulation" / "geometry.py"


def _load_reference():
    """Import the simulator's original geometry.py directly from its path."""
    spec = importlib.util.spec_from_file_location("_ref_geometry", SIM_GEOMETRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dlt(cal, dets):
    """Minimal DLT, so this test exercises the calibration and nothing else."""
    rows = []
    for name, (u, v) in dets.items():
        P = cal[name].P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    return X[:3] / X[3]


# --------------------------------------------------------------------------
# Equivalence with the implementation being replaced
# --------------------------------------------------------------------------

@pytest.mark.skipif(not SIM_GEOMETRY.exists(),
                    reason=f"simulator reference not found at {SIM_GEOMETRY}")
def test_matches_hardcoded_simulator_geometry():
    """Config-driven projection matrices must equal the hard-coded originals."""
    ref = _load_reference()

    assert set(SITE.cam_names) == set(ref.CAMS), "camera set differs"

    np.testing.assert_allclose(SITE["cam_ne"].K, ref.K, rtol=0, atol=1e-12,
                               err_msg="intrinsics differ")

    for name in ref.CAMS:
        np.testing.assert_allclose(
            SITE[name].P, ref.PROJ[name], rtol=0, atol=1e-9,
            err_msg=f"projection matrix differs for {name}")

    # marker_dz deliberately DIVERGES from the reference. The reference took 0.18
    # from the vehicle model file, where it is the marker's height above its parent
    # link; but the pose everything else reports is the model origin, which sits
    # lower. Measured against the model origin in a running simulation the offset
    # is 0.4333 m, constant across a 2.26 m climb. See config/factory.yaml.
    assert ref.MARKER_DZ == pytest.approx(0.18), (
        "reference changed; re-check which frame its offset is relative to")
    assert SITE.marker_dz == pytest.approx(0.4333)


@pytest.mark.skipif(not SIM_GEOMETRY.exists(),
                    reason=f"simulator reference not found at {SIM_GEOMETRY}")
def test_matches_reference_on_real_projections():
    """Equal matrices should mean equal pixels; check end to end, in pixels."""
    ref = _load_reference()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-9, -9, 0.2], [9, 9, 4.4], size=(200, 3))

    for X in pts:
        for name in SITE.cam_names:
            got = SITE[name].project(X)
            h = ref.PROJ[name] @ np.append(X, 1.0)
            if h[2] <= 1e-9:
                continue
            expect = (h[0] / h[2], h[1] / h[2])
            assert got is not None
            assert got[0] == pytest.approx(expect[0], abs=1e-6)
            assert got[1] == pytest.approx(expect[1], abs=1e-6)


# --------------------------------------------------------------------------
# Internal consistency: project then triangulate must recover the point
# --------------------------------------------------------------------------

def test_project_then_triangulate_recovers_point():
    """Round-trip through the geometry with no noise should be near-exact."""
    rng = np.random.default_rng(7)
    pts = rng.uniform([-8, -8, 0.5], [8, 8, 4.0], size=(300, 3))
    worst = 0.0
    for X in pts:
        dets = {n: uv for n in SITE.cam_names
                if (uv := SITE[n].project(X)) is not None}
        assert len(dets) >= SITE.min_views
        worst = max(worst, float(np.linalg.norm(dlt(SITE, dets) - X)))
    assert worst < 1e-9, f"round-trip error {worst:.3e} m is too large"


def test_two_views_suffice():
    """Any camera pair must determine a point; occlusion tolerance rests on it."""
    X = np.array([1.5, -2.0, 2.4])
    names = SITE.cam_names
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = {n: SITE[n].project(X) for n in (names[i], names[j])}
            assert all(v is not None for v in pair.values())
            assert np.linalg.norm(dlt(SITE, pair) - X) < 1e-8, f"{names[i]}+{names[j]}"


def test_all_cameras_see_the_room_centre():
    """Sanity: the cameras are actually aimed at the room, not away from it."""
    for n in SITE.cam_names:
        assert SITE[n].in_view([0.0, 0.0, 2.5]), f"{n} cannot see the room centre"


def test_camera_behind_returns_none():
    """A point behind the camera must be rejected, not wrapped around."""
    cam = SITE["cam_ne"]
    behind = cam.position + (cam.R.T @ np.array([0.0, 0.0, -5.0]))
    assert cam.project(behind) is None


# --------------------------------------------------------------------------
# Config surface
# --------------------------------------------------------------------------

def test_site_contents():
    assert SITE.name == "factory"
    assert len(SITE) == 4
    assert SITE.cam_names == ["cam_ne", "cam_nw", "cam_sw", "cam_se"]
    assert SITE.marker_dz == pytest.approx(0.4333)
    assert SITE.reproj_threshold_px == pytest.approx(25.0)
    assert SITE.min_views == 2
    for n in SITE.cam_names:
        assert SITE[n].resolution == (640, 360)


def test_fov_intrinsics_scale_with_resolution():
    """The old code took resolution from an environment variable, so a mismatch
    silently corrupted the focal length. Resolution now lives beside the model."""
    K640 = intrinsic_matrix({"model": "fov", "hfov": 1.2}, [640, 360])
    K1280 = intrinsic_matrix({"model": "fov", "hfov": 1.2}, [1280, 720])
    assert K1280[0, 0] == pytest.approx(2 * K640[0, 0])
    assert K1280[0, 2] == pytest.approx(640.0)
    assert K1280[1, 2] == pytest.approx(360.0)


def test_pinhole_intrinsics_for_real_cameras():
    K = intrinsic_matrix(
        {"model": "pinhole", "fx": 500.0, "fy": 501.0, "cx": 310.0, "cy": 175.0},
        [640, 360])
    assert (K[0, 0], K[1, 1], K[0, 2], K[1, 2]) == (500.0, 501.0, 310.0, 175.0)


def test_pinhole_defaults_principal_point_to_centre():
    K = intrinsic_matrix({"model": "pinhole", "fx": 500.0}, [640, 360])
    assert (K[1, 1], K[0, 2], K[1, 2]) == (500.0, 320.0, 180.0)


def test_rpy_to_R_is_a_rotation():
    R = rpy_to_R(0.1, 0.2, 0.3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_axis_conventions_are_rotations():
    for name, M in AXIS_CONVENTIONS.items():
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12, err_msg=name)
        assert np.linalg.det(M) == pytest.approx(1.0), name


# --------------------------------------------------------------------------
# Bad configuration must fail loudly, at load time
# --------------------------------------------------------------------------

def _doc(**over):
    doc = {
        "name": "t",
        "defaults": {"resolution": [640, 360],
                     "intrinsics": {"model": "fov", "hfov": 1.2},
                     "axes": "gz_camera"},
        "cameras": {"a": {"position": [1, 1, 1], "rpy": [0, 0, 0]}},
    }
    doc.update(over)
    return doc


@pytest.mark.parametrize("over, match", [
    ({"cameras": {}}, "no cameras"),
    ({"cameras": {"a": {"position": [1, 1, 1], "rpy": [0, 0, 0],
                        "axes": "nonsense"}}}, "unknown axes"),
    ({"cameras": {"a": {"position": [1, 1], "rpy": [0, 0, 0]}}}, "position must be"),
    ({"cameras": {"a": {"position": [1, 1, 1], "rpy": [0, 0, 0],
                        "resolution": [640]}}}, "resolution must be"),
])
def test_invalid_config_raises(over, match):
    with pytest.raises(ValueError, match=match):
        Calibration.from_dict(_doc(**over))


@pytest.mark.parametrize("hfov", [0.0, -1.0, math.pi, 4.0])
def test_impossible_fov_rejected(hfov):
    with pytest.raises(ValueError, match="hfov"):
        intrinsic_matrix({"model": "fov", "hfov": hfov}, [640, 360])


def test_unknown_intrinsics_model_rejected():
    with pytest.raises(ValueError, match="unknown intrinsics model"):
        intrinsic_matrix({"model": "fisheye"}, [640, 360])
