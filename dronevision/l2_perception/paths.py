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


#: What each runtime loads. Lets a sweep name one model stem and iterate runtimes.
RUNTIME_SUFFIX = {"ultralytics": ".pt", "onnx": ".onnx", "tflite": ".tflite", "executorch": ".pte"}


def resolve_model(name_or_path, runtime="ultralytics"):
    """Find the artifact a runtime should load.

    Accepts an explicit path (used as-is), or a bare stem such as
    ``drone_yolo26n_v4``, which is completed with the suffix the runtime needs. That is
    what lets a benchmark sweep runtimes and quantizations without any code knowing which
    file extension belongs to which engine.
    """
    suffix = RUNTIME_SUFFIX.get(runtime, "")

    if name_or_path:
        p = Path(name_or_path)
        if p.exists():
            return p
        if p.suffix:                       # caller named a concrete file
            return model_path(p)
        return model_path(p.with_suffix(suffix))

    # Nothing specified: take the default stem and give it the runtime's extension.
    for cand in DEFAULT_YOLO_CANDIDATES:
        p = models_dir() / (Path(cand).stem + suffix)
        if p.exists():
            return p

    have = sorted(x.name for x in models_dir().glob("*") if x.suffix == suffix) or "none"
    raise FileNotFoundError(
        f"no {suffix or '<any>'} model for runtime {runtime!r} in {models_dir()}\n"
        f"  available: {have}\n"
        f"  set YOLO_MODEL, or pass an explicit path, or export one with tools/export.py")


# Retained for call sites that build paths by string concatenation.
MODELS_DIR = str(models_dir())
