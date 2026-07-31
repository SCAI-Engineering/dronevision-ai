"""Where model weights live.

Weights ship with the repository in `models/`, so a clone is immediately runnable
with no download step. `DRONEVISION_MODELS` overrides the directory — which is
how a Raspberry Pi points at weights kept on a larger volume, or how a benchmark
sweeps several export variants of the same network.
"""
import os
from pathlib import Path

#: Preference order for the default detector network. The first that exists wins,
#: so a repository carrying only some of these still works.
DEFAULT_YOLO_CANDIDATES = (
    "drone_yolo26n_v4.pt",
    "drone_yolo26n.pt",
)


def models_dir():
    """Directory holding model weights."""
    env = os.environ.get("DRONEVISION_MODELS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"


def model_path(name, required=True):
    """Resolve a weight file by name. Absolute paths pass through untouched."""
    p = Path(name)
    if p.is_absolute():
        if required and not p.exists():
            raise FileNotFoundError(f"model not found: {p}")
        return p
    p = models_dir() / name
    if required and not p.exists():
        raise FileNotFoundError(
            f"model not found: {p}\n"
            f"(set DRONEVISION_MODELS to point elsewhere; "
            f"available: {sorted(x.name for x in models_dir().glob('*')) or 'none'})")
    return p


def default_yolo():
    """Path to the default detector network, or None if no candidate is present."""
    for cand in DEFAULT_YOLO_CANDIDATES:
        p = models_dir() / cand
        if p.exists():
            return p
    return None


# Retained for call sites that build paths by string concatenation.
MODELS_DIR = str(models_dir())
