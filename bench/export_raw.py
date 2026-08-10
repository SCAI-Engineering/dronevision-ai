#!/usr/bin/env python3
"""Export the trained YOLO26 checkpoint without its end-to-end selection head.

The raw decoded output is suitable for static quantization and Vela because TopK and final
detection selection stay outside the graph.  Install the exporter dependencies in an
isolated environment, then run:

    uv pip install 'ultralytics==8.4.104' 'onnx>=1.17,<2' 'onnxslim>=0.1.82'
    python -m bench.export_raw
"""
import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path


EXPECTED_ULTRALYTICS = "8.4.104"
DEFAULT_SOURCE = "models/drone_yolo26n_v4.pt"
DEFAULT_OUTPUT = "models/drone_yolo26n_v4_raw.onnx"


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [dim.dim_value if dim.HasField("dim_value") else None for dim in dims]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--allow-version-mismatch", action="store_true",
        help="allow an exporter other than the checkpoint's original Ultralytics version",
    )
    args = parser.parse_args(argv)

    try:
        import onnx
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            f"missing export dependency: {exc}. Use the pinned install command in this module.")

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS and not args.allow_version_mismatch:
        raise SystemExit(
            f"Ultralytics {ultralytics.__version__} is installed, but the source checkpoint "
            f"was exported with {EXPECTED_ULTRALYTICS}; use that version or explicitly pass "
            "--allow-version-mismatch")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise SystemExit(f"source checkpoint does not exist: {source}")
    if source == output:
        raise SystemExit("source and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)

    # Ultralytics derives the output name from the checkpoint path. A temporary copy gives
    # the export its requested stem without renaming or overwriting the immutable source.
    with tempfile.TemporaryDirectory(prefix="dronevision-raw-export-") as tmp:
        temporary_source = Path(tmp) / f"{output.stem}.pt"
        shutil.copyfile(source, temporary_source)
        model = YOLO(temporary_source)
        checkpoint_version = str((getattr(model, "ckpt", None) or {}).get("version", ""))
        if (
            checkpoint_version
            and checkpoint_version != ultralytics.__version__
            and not args.allow_version_mismatch
        ):
            raise SystemExit(
                f"checkpoint says Ultralytics {checkpoint_version}, but "
                f"{ultralytics.__version__} is installed")

        exported = Path(model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
            nms=False,
            end2end=False,
            device="cpu",
        ))
        shutil.copyfile(exported, output)

    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    input_shapes = {value.name: tensor_shape(value) for value in graph.graph.input}
    output_shapes = {value.name: tensor_shape(value) for value in graph.graph.output}
    expected_input = [1, 3, args.imgsz, args.imgsz]
    expected_output = [1, 5, sum((args.imgsz // stride) ** 2 for stride in (8, 16, 32))]
    if list(input_shapes.values()) != [expected_input]:
        output.unlink(missing_ok=True)
        raise SystemExit(f"unexpected ONNX input shape: {input_shapes}, expected {expected_input}")
    if list(output_shapes.values()) != [expected_output]:
        output.unlink(missing_ok=True)
        raise SystemExit(f"unexpected ONNX output shape: {output_shapes}, expected {expected_output}")

    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update({
        "source_file": source.name,
        "source_sha256": source_hash,
        "export_purpose": "raw decoded YOLO head for INT8 and Ethos-U65",
        "end2end": "False",
    })
    onnx.helper.set_model_props(graph, metadata)
    onnx.save(graph, output)
    onnx.checker.check_model(onnx.load(output))

    print(f"source: {source}")
    print(f"source sha256: {source_hash}")
    print(f"output: {output}")
    print(f"output sha256: {sha256(output)}")
    print(f"input: {input_shapes}")
    print(f"output: {output_shapes}")
    print(f"ultralytics: {ultralytics.__version__}, opset: {args.opset}, end2end: False")


if __name__ == "__main__":
    main()
