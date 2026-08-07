#!/usr/bin/env python3
"""Materialize representative calibration images from data/corpus/quantization_split.json.

Extracts exactly the 256 calibration JPEG files from frames.zip into ignored data/scratch/
storage without decoding or re-encoding bytes. Generates image manifests and a dataset YAML
for model export and INT8 quantization tools.

Usage:
    python -m bench.materialize_quant_data
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from dronevision.io.sources.replay import FRAMES_FILE, MANIFEST_FILE, META_FILE


DEFAULT_SPLIT_MANIFEST = "data/corpus/quantization_split.json"
DEFAULT_OUTPUT_DIR = "data/scratch/quantization_calibration"


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus_hashes(corpus_dir, expected_hashes):
    corpus_dir = Path(corpus_dir)
    for filename in (META_FILE, MANIFEST_FILE, FRAMES_FILE):
        file_path = corpus_dir / filename
        if not file_path.is_file():
            raise RuntimeError(f"required corpus file missing: {file_path}")
        actual = sha256(file_path)
        expected = expected_hashes.get(filename)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    split_path = Path(args.split_manifest).resolve()
    if not split_path.is_file():
        raise SystemExit(f"quantization split manifest missing: {split_path}")

    with open(split_path, "r", encoding="utf-8") as fh:
        split_data = json.load(fh)

    corpus_info = split_data.get("corpus", {})
    corpus_dir = split_path.parent.resolve()
    verify_corpus_hashes(corpus_dir, corpus_info.get("sha256", {}))

    calibration = split_data.get("calibration", {})
    calibration_images = calibration.get("images", [])
    expected_image_count = calibration.get("image_count", 256)
    expected_frame_set_count = calibration.get("frame_set_count", 64)

    if len(calibration_images) != expected_image_count:
        raise SystemExit(
            f"expected {expected_image_count} calibration images in split, found {len(calibration_images)}")

    frame_set_ids = set(img["frame_set_id"] for img in calibration_images)
    if len(frame_set_ids) != expected_frame_set_count:
        raise SystemExit(
            f"expected {expected_frame_set_count} unique frame sets, found {len(frame_set_ids)}")

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames_zip_path = corpus_dir / FRAMES_FILE
    extracted_paths = []

    with zipfile.ZipFile(frames_zip_path, "r") as zf:
        for item in calibration_images:
            zip_member = item["zip_member"]
            target_path = images_dir / zip_member
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Read exact bytes from ZIP and write to target
            raw_bytes = zf.read(zip_member)
            target_path.write_bytes(raw_bytes)
            extracted_paths.append(target_path)

    if len(extracted_paths) != expected_image_count:
        raise SystemExit(
            f"extracted {len(extracted_paths)} files, expected {expected_image_count}")

    # Write file lists for quantization calibration
    list_path = output_dir / "calib_images.txt"
    list_path.write_text(
        "\n".join(str(p) for p in sorted(extracted_paths)) + "\n",
        encoding="utf-8"
    )

    # Write minimal dataset YAML (Ultralytics compatible)
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = (
        f"# Representative dataset for INT8 quantization\n"
        f"path: {images_dir.as_posix()}\n"
        f"train: {images_dir.as_posix()}\n"
        f"val: {images_dir.as_posix()}\n"
        f"names:\n"
        f"  0: drone\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Write manifest index linking frame set, camera, path and byte size
    manifest_index = []
    for item in calibration_images:
        p = images_dir / item["zip_member"]
        manifest_index.append({
            "frame_set_id": item["frame_set_id"],
            "camera": item["camera"],
            "zip_member": item["zip_member"],
            "extracted_path": str(p),
            "size_bytes": p.stat().st_size,
        })
    (output_dir / "extracted_manifest.json").write_text(
        json.dumps(manifest_index, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Verified corpus SHA-256 hashes successfully.")
    print(f"Extracted {len(extracted_paths)} calibration JPEGs ({expected_frame_set_count} frame sets) to {images_dir}.")
    print(f"Wrote image list: {list_path}")
    print(f"Wrote dataset YAML: {yaml_path}")
    print(f"Wrote extracted manifest: {output_dir / 'extracted_manifest.json'}")


if __name__ == "__main__":
    main()
