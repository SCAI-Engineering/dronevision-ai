#!/usr/bin/env python3
"""Export YOLO26 raw-head model to FP32 TFLite and full INT8 TFLite.

Steps 3.3 and 3.4 of the i.MX93 Ethos-U65 plan:
1. Export raw FP32 TFLite (no NMS/end2end head, static 320x320, batch 1).
2. Export full INT8 TFLite using the 256 materialized calibration JPEGs.
3. Record input/output shapes, dtypes, scales, zero points, and SHA-256 hashes.

Usage:
    python -m bench.export_tflite --mode fp32
    python -m bench.export_tflite --mode int8
    python -m bench.export_tflite --mode both
"""
import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from dronevision.l2_perception.runtimes.tflite_rt import _get_interpreter


DEFAULT_SOURCE = "models/drone_yolo26n_v4.pt"
OUTPUT_FP32 = "models/drone_yolo26n_v4_raw_fp32.tflite"
OUTPUT_INT8 = "models/drone_yolo26n_v4_raw_int8.tflite"
DEFAULT_DATASET = "data/scratch/quantization_calibration/dataset.yaml"


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_tflite(model_path):
    interpreter = _get_interpreter(model_path, threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return {
        "path": str(model_path),
        "sha256": sha256(model_path),
        "size_bytes": Path(model_path).stat().st_size,
        "input": {
            "name": in_det["name"],
            "shape": list(in_det["shape"]),
            "dtype": str(in_det["dtype"]),
            "quantization": in_det.get("quantization", (0.0, 0)),
        },
        "output": {
            "name": out_det["name"],
            "shape": list(out_det["shape"]),
            "dtype": str(out_det["dtype"]),
            "quantization": out_det.get("quantization", (0.0, 0)),
        },
    }


def export_fp32(source_pt, output_path, imgsz=320):
    from ultralytics import YOLO

    source = Path(source_pt).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dronevision-tflite-fp32-") as tmp:
        tmp_pt = Path(tmp) / f"{output.stem}.pt"
        shutil.copyfile(source, tmp_pt)
        model = YOLO(tmp_pt)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
            for m in model.model.modules():
                m.eval()

        res = model.export(
            format="litert",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            opset=12,
            nms=False,
            end2end=False,
            device="cpu",
            int8=False,
        )

        res_path = Path(res) if isinstance(res, (str, Path)) else Path(str(res))
        if not res_path.is_file():
            # Ultralytics may name it stem_saved_model/stem_float32.tflite or similar
            possible = list(res_path.parent.glob("*.tflite")) + list(res_path.glob("*.tflite"))
            if possible:
                res_path = possible[0]
            else:
                raise RuntimeError(f"could not find exported tflite file from {res}")

        shutil.copyfile(res_path, output)

    return inspect_tflite(output)


def export_int8(source_pt, output_path, dataset_yaml, imgsz=320):
    from ultralytics import YOLO

    source = Path(source_pt).resolve()
    output = Path(output_path).resolve()
    dataset = Path(dataset_yaml).resolve()
    if not dataset.is_file():
        raise SystemExit(f"calibration dataset YAML missing: {dataset}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dronevision-tflite-int8-") as tmp:
        tmp_pt = Path(tmp) / f"{output.stem}.pt"
        shutil.copyfile(source, tmp_pt)
        model = YOLO(tmp_pt)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
            for m in model.model.modules():
                m.eval()

        res = model.export(
            format="litert",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            nms=False,
            end2end=False,
            device="cpu",
            int8=True,
            data=str(dataset),
        )

        res_path = Path(res) if isinstance(res, (str, Path)) else Path(str(res))
        if not res_path.is_file():
            possible = list(res_path.parent.glob("*int8*.tflite")) + list(res_path.parent.glob("*.tflite")) + list(res_path.glob("*.tflite"))
            if possible:
                res_path = possible[0]
            else:
                raise RuntimeError(f"could not find exported INT8 tflite file from {res}")

        shutil.copyfile(res_path, output)

    return inspect_tflite(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-fp32", default=OUTPUT_FP32)
    parser.add_argument("--output-int8", default=OUTPUT_INT8)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--mode", choices=("fp32", "int8", "both"), default="both")
    args = parser.parse_args(argv)

    if args.mode in ("fp32", "both"):
        print("=== Exporting FP32 TFLite (Step 3.3) ===")
        info_fp32 = export_fp32(args.source, args.output_fp32, imgsz=args.imgsz)
        print("FP32 Inspection:")
        print(f"  Path: {info_fp32['path']}")
        print(f"  SHA-256: {info_fp32['sha256']}")
        print(f"  Size: {info_fp32['size_bytes']} bytes")
        print(f"  Input: {info_fp32['input']}")
        print(f"  Output: {info_fp32['output']}\n")

    if args.mode in ("int8", "both"):
        print("=== Exporting Full INT8 TFLite (Step 3.4) ===")
        info_int8 = export_int8(args.source, args.output_int8, args.dataset, imgsz=args.imgsz)
        print("INT8 Inspection:")
        print(f"  Path: {info_int8['path']}")
        print(f"  SHA-256: {info_int8['sha256']}")
        print(f"  Size: {info_int8['size_bytes']} bytes")
        print(f"  Input: {info_int8['input']}")
        print(f"  Output: {info_int8['output']}\n")


if __name__ == "__main__":
    main()
