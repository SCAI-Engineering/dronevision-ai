"""Site calibration: load camera intrinsics and extrinsics from configuration.

Replaces what used to be module-level constants — a hard-coded camera dictionary,
a focal length derived from an environment variable, and a bare 3x3 axis-swap
literal. Those made a second deployment site a code edit, and made a resolution
change silently wrong if you forgot to set a matching environment variable.

A site is now one YAML file (see `config/factory.yaml`) producing one
`Calibration` object, which owns the projection matrices everything downstream
needs. Real cameras with a real calibration are a different file, not a patch.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Camera axis conventions, as world->camera rotations applied *after* the pose.
#
# `gz_camera`: a simulated gz camera looks along its own +x, with +y to the left
# and +z up. The projection maths below assumes the standard optical convention
# (+x right, +y down, +z along the view direction), so the axes must be permuted.
# `optical`: already in the optical convention; no adaptation needed.
AXIS_CONVENTIONS = {
    "gz_camera": np.array([[0, -1, 0],
                           [0, 0, -1],
                           [1, 0, 0]], float),
    "optical": np.eye(3),
}


def rpy_to_R(roll, pitch, yaw):
    """Body->world rotation from roll/pitch/yaw, applied Z then Y then X."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    return Rz @ Ry @ Rx


def intrinsic_matrix(spec, resolution):
    """Build K (3x3) from an intrinsics spec and a [width, height] resolution.

    Two models, chosen by the ``model`` key:

    ``fov``      focal length from horizontal field of view, square pixels,
                 principal point at the image centre. How simulated cameras are
                 specified. Because fx depends on width, K is always consistent
                 with the resolution actually in use.
    ``pinhole``  fx, fy, cx, cy given directly, as a real calibration produces.
                 Defaults place the principal point at the centre if omitted.
    """
    w, h = int(resolution[0]), int(resolution[1])
    model = spec.get("model", "fov")

    if model == "fov":
        hfov = float(spec["hfov"])
        if not 0.0 < hfov < math.pi:
            raise ValueError(f"hfov must be in (0, pi) radians, got {hfov}")
        fx = fy = (w / 2.0) / math.tan(hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0
    elif model == "pinhole":
        fx = float(spec["fx"])
        fy = float(spec.get("fy", fx))
        cx = float(spec.get("cx", w / 2.0))
        cy = float(spec.get("cy", h / 2.0))
    else:
        raise ValueError(
            f"unknown intrinsics model {model!r}; expected 'fov' or 'pinhole'")

    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], float)


@dataclass(frozen=True)
class Camera:
    """One fixed camera: where it is, what it sees, how to project into it."""

    name: str
    resolution: tuple          # (width, height) px
    K: np.ndarray              # 3x3 intrinsics
    position: np.ndarray       # (3,) world ENU, metres
    rpy: tuple                 # (roll, pitch, yaw) radians
    R: np.ndarray              # 3x3 world->camera-optical rotation
    P: np.ndarray              # 3x4 projection, world ENU -> homogeneous pixels

    def project(self, xyz):
        """World point -> (u, v) pixels. Returns None if behind the camera."""
        h = self.P @ np.append(np.asarray(xyz, float), 1.0)
        if h[2] <= 1e-9:
            return None
        return (h[0] / h[2], h[1] / h[2])

    def in_view(self, xyz, margin=0.0):
        """True if `xyz` projects inside the image (optionally inset by `margin`)."""
        uv = self.project(xyz)
        if uv is None:
            return False
        w, h = self.resolution
        return (margin <= uv[0] <= w - margin) and (margin <= uv[1] <= h - margin)


@dataclass(frozen=True)
class Calibration:
    """Everything about one deployment site that the geometry needs."""

    name: str
    cameras: dict                    # name -> Camera, insertion-ordered
    marker_dz: float = 0.0
    reproj_threshold_px: float = 25.0
    min_views: int = 2
    raw: dict = None                 # the parsed document, for anything not modelled

    # -- convenience views used throughout the pipeline ---------------------

    @property
    def cam_names(self):
        """Camera names in configuration order. The scheduler rotates over this."""
        return list(self.cameras)

    @property
    def P(self):
        """``{name: 3x4 projection matrix}``."""
        return {n: c.P for n, c in self.cameras.items()}

    def __getitem__(self, name):
        return self.cameras[name]

    def __contains__(self, name):
        return name in self.cameras

    def __len__(self):
        return len(self.cameras)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path):
        """Read a site YAML file and build the calibration."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.from_dict(doc, source=str(path))

    @classmethod
    def from_dict(cls, doc, source="<dict>"):
        defaults = doc.get("defaults", {}) or {}
        cam_specs = doc.get("cameras") or {}
        if not cam_specs:
            raise ValueError(f"{source}: no cameras defined")

        cameras = {}
        for name, spec in cam_specs.items():
            spec = spec or {}
            resolution = tuple(spec.get("resolution", defaults.get("resolution")))
            if len(resolution) != 2:
                raise ValueError(f"{source}: {name}: resolution must be [w, h]")

            intr = spec.get("intrinsics", defaults.get("intrinsics"))
            if intr is None:
                raise ValueError(f"{source}: {name}: no intrinsics and no default")
            K = intrinsic_matrix(intr, resolution)

            axes = spec.get("axes", defaults.get("axes", "optical"))
            if axes not in AXIS_CONVENTIONS:
                raise ValueError(
                    f"{source}: {name}: unknown axes convention {axes!r}; "
                    f"expected one of {sorted(AXIS_CONVENTIONS)}")

            position = np.asarray(spec["position"], float)
            if position.shape != (3,):
                raise ValueError(f"{source}: {name}: position must be [x, y, z]")
            rpy = tuple(float(v) for v in spec["rpy"])
            if len(rpy) != 3:
                raise ValueError(f"{source}: {name}: rpy must be [roll, pitch, yaw]")

            # World->camera rotation: invert the camera's world orientation, then
            # permute into the optical convention.
            R = AXIS_CONVENTIONS[axes] @ rpy_to_R(*rpy).T
            # P = K [R | t] with t = -R @ position, i.e. the world origin
            # expressed in camera coordinates.
            P = K @ np.hstack([R, (-R @ position).reshape(3, 1)])

            cameras[name] = Camera(name=name, resolution=resolution, K=K,
                                   position=position, rpy=rpy, R=R, P=P)

        target = doc.get("target", {}) or {}
        tri = doc.get("triangulation", {}) or {}
        return cls(
            name=doc.get("name", "unnamed"),
            cameras=cameras,
            marker_dz=float(target.get("marker_dz", 0.0)),
            reproj_threshold_px=float(tri.get("reproj_threshold_px", 25.0)),
            min_views=int(tri.get("min_views", 2)),
            raw=doc,
        )


def default_config_path(name="factory"):
    """Path to a bundled site config, resolved relative to the repository."""
    return Path(__file__).resolve().parents[2] / "config" / f"{name}.yaml"


def load_site(name_or_path="factory"):
    """Load a site by bundled name (``"factory"``) or by explicit path."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return Calibration.load(p)
    return Calibration.load(default_config_path(str(name_or_path)))
