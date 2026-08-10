#!/usr/bin/env python3
"""Export YOLO26 raw-head model to FP32 and full INT8 TFLite variants.

Steps 3.3 and 3.4 of the i.MX93 Ethos-U65 plan:
1. Export raw FP32 TFLite (no NMS/end2end head, static 320x320, batch 1).
2. Export full INT8 TFLite using the 256 materialized calibration JPEGs.
3. Record input/output shapes, dtypes, scales, zero points, and SHA-256 hashes.
4. Optionally expose the full-INT8 graph's genuine integer boundaries.

Usage:
    python -m bench.export_tflite --mode fp32
    python -m bench.export_tflite --mode int8
    python -m bench.export_tflite --mode int8-io
    python -m bench.export_tflite --mode both
"""
import argparse
import hashlib
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

from dronevision.l2_perception.runtimes.tflite_rt import _get_interpreter


DEFAULT_SOURCE = "models/drone_yolo26n_v4.pt"
OUTPUT_FP32 = "models/drone_yolo26n_v4_raw_fp32.tflite"
OUTPUT_INT8 = "models/drone_yolo26n_v4_raw_int8.tflite"
OUTPUT_INT8_IO = "models/drone_yolo26n_v4_raw_int8_io.tflite"
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
            "dtype": np.dtype(in_det["dtype"]).name,
            "quantization": in_det.get("quantization", (0.0, 0)),
        },
        "output": {
            "name": out_det["name"],
            "shape": list(out_det["shape"]),
            "dtype": np.dtype(out_det["dtype"]).name,
            "quantization": out_det.get("quantization", (0.0, 0)),
        },
    }


def export_int8_integer_io(source_int8, output_path):
    """Expose the existing full-INT8 graph's internal integer boundaries.

    Ultralytics' LiteRT exporter intentionally keeps public input/output tensors in
    FP32.  Ethos-U then receives a three-node graph:

        QUANTIZE (CPU) -> ethos-u (NPU) -> DEQUANTIZE (CPU)

    The tensors immediately after/before those boundary nodes are already genuine
    calibrated INT8 tensors. Rewriting the subgraph inputs/outputs and removing
    only those two conversion operators preserves every weight, scale, zero point,
    and operator in the Vela-compatible quantized graph.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    source = Path(source_int8).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise SystemExit(f"full INT8 TFLite source missing: {source}")

    # Mutate the FlatBuffer vectors in place. The LiteRT export stores large
    # constants as external buffers, so unpacking/repacking the model would
    # invalidate their absolute offsets.
    model_bytes = bytearray(source.read_bytes())
    model = schema.Model.GetRootAsModel(model_bytes, 0)
    if model.SubgraphsLength() != 1:
        raise RuntimeError(f"expected one TFLite subgraph, found {model.SubgraphsLength()}")
    graph = model.Subgraphs(0)
    if graph.InputsLength() != 1 or graph.OutputsLength() != 1 or graph.OperatorsLength() < 3:
        raise RuntimeError("unexpected TFLite boundary structure")

    first, last = graph.Operators(0), graph.Operators(graph.OperatorsLength() - 1)
    first_code = model.OperatorCodes(first.OpcodeIndex()).BuiltinCode()
    last_code = model.OperatorCodes(last.OpcodeIndex()).BuiltinCode()
    if first_code != schema.BuiltinOperator.QUANTIZE:
        raise RuntimeError(f"first operator is {first_code}, expected QUANTIZE")
    if last_code != schema.BuiltinOperator.DEQUANTIZE:
        raise RuntimeError(f"last operator is {last_code}, expected DEQUANTIZE")
    if first.OutputsLength() != 1 or last.InputsLength() != 1:
        raise RuntimeError("unexpected quantize/dequantize boundary arity")

    integer_input = first.Outputs(0)
    integer_output = last.Inputs(0)
    if graph.Tensors(integer_input).Type() != schema.TensorType.INT8:
        raise RuntimeError("tensor after input QUANTIZE is not INT8")
    if graph.Tensors(integer_output).Type() != schema.TensorType.INT8:
        raise RuntimeError("tensor before output DEQUANTIZE is not INT8")

    graph.InputsAsNumpy()[0] = integer_input
    graph.OutputsAsNumpy()[0] = integer_output

    # A FlatBuffer vector of tables stores relative offsets. Compact references
    # to operators 1..N-2 and shorten the vector, leaving the model body intact.
    old_count = graph.OperatorsLength()
    kept_positions = [graph.Operators(i)._tab.Pos for i in range(1, old_count - 1)]
    vector_start = graph._tab.Vector(graph._tab.Offset(10))
    struct.pack_into("<I", model_bytes, vector_start - 4, len(kept_positions))
    for index, target in enumerate(kept_positions):
        slot = vector_start + index * 4
        struct.pack_into("<I", model_bytes, slot, target - slot)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model_bytes)

    # Inspect through the schema rather than allocating an interpreter: a
    # Vela-compiled model contains the custom ``ethos-u`` op, which cannot be
    # prepared on the export host without the hardware delegate.
    rewritten = schema.Model.GetRootAsModel(output.read_bytes(), 0).Subgraphs(0)

    def tensor_info(index):
        tensor = rewritten.Tensors(index)
        quant = tensor.Quantization()
        scale = quant.Scale(0) if quant and quant.ScaleLength() else 0.0
        zero_point = quant.ZeroPoint(0) if quant and quant.ZeroPointLength() else 0
        return {
            "name": tensor.Name().decode(),
            "shape": [tensor.Shape(i) for i in range(tensor.ShapeLength())],
            "dtype": "int8" if tensor.Type() == schema.TensorType.INT8 else str(tensor.Type()),
            "quantization": (scale, zero_point),
        }

    info = {
        "path": str(output),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "input": tensor_info(rewritten.Inputs(0)),
        "output": tensor_info(rewritten.Outputs(0)),
    }
    if info["input"]["dtype"] != "int8" or info["output"]["dtype"] != "int8":
        raise RuntimeError(f"integer-I/O export failed tensor contract: {info}")
    return info


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
    parser.add_argument("--output-int8-io", default=OUTPUT_INT8_IO)
    parser.add_argument(
        "--source-int8-io",
        default=None,
        help="INT8 model whose FP32 boundary ops are removed (defaults to --output-int8); may be a Vela model",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument(
        "--mode", choices=("fp32", "int8", "int8-io", "both", "all"), default="both"
    )
    args = parser.parse_args(argv)

    if args.mode in ("fp32", "both", "all"):
        print("=== Exporting FP32 TFLite (Step 3.3) ===")
        info_fp32 = export_fp32(args.source, args.output_fp32, imgsz=args.imgsz)
        print("FP32 Inspection:")
        print(f"  Path: {info_fp32['path']}")
        print(f"  SHA-256: {info_fp32['sha256']}")
        print(f"  Size: {info_fp32['size_bytes']} bytes")
        print(f"  Input: {info_fp32['input']}")
        print(f"  Output: {info_fp32['output']}\n")

    if args.mode in ("int8", "both", "all"):
        print("=== Exporting Full INT8 TFLite (Step 3.4) ===")
        info_int8 = export_int8(args.source, args.output_int8, args.dataset, imgsz=args.imgsz)
        print("INT8 Inspection:")
        print(f"  Path: {info_int8['path']}")
        print(f"  SHA-256: {info_int8['sha256']}")
        print(f"  Size: {info_int8['size_bytes']} bytes")
        print(f"  Input: {info_int8['input']}")
        print(f"  Output: {info_int8['output']}\n")

    if args.mode in ("int8-io", "all"):
        print("=== Exporting Full INT8 TFLite with INT8 I/O ===")
        info_int8_io = export_int8_integer_io(
            args.source_int8_io or args.output_int8,
            args.output_int8_io,
        )
        print("INT8 I/O Inspection:")
        print(f"  Path: {info_int8_io['path']}")
        print(f"  SHA-256: {info_int8_io['sha256']}")
        print(f"  Size: {info_int8_io['size_bytes']} bytes")
        print(f"  Input: {info_int8_io['input']}")
        print(f"  Output: {info_int8_io['output']}\n")


if __name__ == "__main__":
    main()
