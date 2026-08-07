"""Inference runtimes — which engine executes the detection network.

Orthogonal to the detector: `--detector yolo --runtime onnx` picks *what* to find and
*what runs it* independently. That separation is the point of the project, since the same
weights are what get compared across engines, quantizations and Arm cores.

NOTHING IS IMPORTED HERE. The registry holds module paths as strings and resolves them
inside `make_runtime`, so importing this package never drags in PyTorch or ONNX Runtime.
A Raspberry Pi is expected to have exactly one of them installed, and `tests/test_boundary.py`
fails the build if a heavy dependency reaches module scope anywhere under `dronevision/`.
"""
import importlib
import importlib.util

RUNTIMES = {
    "ultralytics": ("dronevision.l2_perception.runtimes.ultralytics_rt",
                    "UltralyticsRuntime", "torch"),
    "onnx": ("dronevision.l2_perception.runtimes.onnx_rt",
             "OnnxRuntime", "ort"),
    "tflite": ("dronevision.l2_perception.runtimes.tflite_rt",
               "TfLiteRuntime", "pi"),
    "executorch": ("dronevision.l2_perception.runtimes.executorch_rt",
                   "ExecuTorchRuntime", "executorch"),
}

#: Convenience spellings. `torch` and `pt` mean the reference path; `ort` means ONNX.
ALIASES = {"torch": "ultralytics", "pt": "ultralytics", "yolo": "ultralytics",
           "ort": "onnx", "onnxruntime": "onnx", "tflite": "tflite", "litert": "tflite",
           "et": "executorch", "pte": "executorch"}

#: Which pip extra provides each runtime, for error messages worth reading.
REQUIRES = {"ultralytics": ("torch", "ultralytics"), "onnx": ("onnxruntime",),
            "tflite": ("tflite_runtime",), "executorch": ("executorch",)}

DEFAULT = "ultralytics"


def canonical(name):
    """Resolve an alias to a registry key, or raise listing what exists."""
    key = (name or DEFAULT).lower()
    key = ALIASES.get(key, key)
    if key not in RUNTIMES:
        raise ValueError(
            f"unknown runtime {name!r}; expected one of "
            f"{sorted(RUNTIMES)} (aliases: {sorted(ALIASES)})")
    return key


def available():
    """``{runtime: importable}`` without importing anything."""
    out = {}
    for key in RUNTIMES:
        if key == "tflite":
            out[key] = any(importlib.util.find_spec(m) is not None for m in ("ai_edge_litert", "tflite_runtime", "tensorflow"))
        else:
            mods = REQUIRES.get(key, ())
            try:
                out[key] = all(importlib.util.find_spec(m) is not None for m in mods)
            except (ImportError, ValueError):
                out[key] = False
    return out


def make_runtime(name=None, **kw):
    """Build a runtime by name. Heavy imports happen here, never at module scope."""
    import os

    key = canonical(name or os.environ.get("RUNTIME") or DEFAULT)
    module_path, class_name, extra = RUNTIMES[key]
    if not available().get(key):
        raise RuntimeError(
            f"runtime {key!r} needs {' + '.join(REQUIRES[key])}, which is not installed.\n"
            f"  pip install -e \".[{extra}]\"")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kw)
