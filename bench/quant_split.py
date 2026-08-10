#!/usr/bin/env python3
"""Build and validate the deterministic INT8 calibration/holdout split.

The corpus stays in its original ZIP.  This manifest records which compressed frames a
converter should decode for representative calibration and which synchronized frame sets
remain outside calibration for the first accuracy check.

    python -m bench.quant_split
    python -m bench.quant_split --calibration-sets 64
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from dronevision.io.sources.replay import FRAMES_FILE, MANIFEST_FILE, META_FILE, read_manifest, read_meta


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = "quantization_split.json"


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(total, count):
    """Return `count` unique indices spanning 0..total-1, with integer rounding.

    The explicit half-up formula is reproducible across languages and avoids Python's
    round-to-even behaviour becoming an accidental part of the dataset definition.
    """
    if total < 1:
        raise ValueError("the corpus has no frame sets")
    if not 1 <= count <= total:
        raise ValueError(f"calibration count must be in 1..{total}, got {count}")
    if count == 1:
        return [0]
    denominator = count - 1
    half = denominator // 2
    selected = [(i * (total - 1) + half) // denominator for i in range(count)]
    if len(set(selected)) != count:
        raise ValueError("selection produced duplicate frame-set indices")
    return selected


def build_split(corpus, calibration_sets=64):
    corpus = Path(corpus)
    meta = read_meta(corpus)
    records = read_manifest(corpus)
    cameras = list(meta["cameras"])
    selected_positions = evenly_spaced_indices(len(records), calibration_sets)
    selected_ids = [records[pos]["i"] for pos in selected_positions]
    selected_id_set = set(selected_ids)

    record_ids = [rec["i"] for rec in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("corpus manifest contains duplicate frame-set IDs")
    if meta.get("frame_sets") != len(records):
        raise ValueError(
            f"meta says {meta.get('frame_sets')} frame sets but manifest has {len(records)}")

    zip_path = corpus / FRAMES_FILE
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())

    calibration_images = []
    referenced = set()
    for rec in records:
        for camera in cameras:
            entry = rec.get("cams", {}).get(camera)
            if entry is None:
                raise ValueError(f"frame set {rec['i']} is missing camera {camera}")
            member = entry["f"]
            if member not in members:
                raise ValueError(f"frame set {rec['i']} references missing ZIP member {member}")
            key = (rec["i"], camera, member)
            if key in referenced:
                raise ValueError(f"duplicate frame reference {key}")
            referenced.add(key)
            if rec["i"] in selected_id_set:
                calibration_images.append({
                    "frame_set_id": rec["i"],
                    "camera": camera,
                    "zip_member": member,
                })

    validation_ids = [frame_id for frame_id in record_ids if frame_id not in selected_id_set]
    timestamps = [records[pos]["t"] for pos in selected_positions]
    expected_calibration_images = calibration_sets * len(cameras)
    if len(calibration_images) != expected_calibration_images:
        raise ValueError(
            f"expected {expected_calibration_images} calibration images, "
            f"found {len(calibration_images)}")

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "INT8 representative calibration and quantization holdout split",
        "corpus": {
            "path": corpus.as_posix(),
            "site": meta.get("site"),
            "frame_sets": len(records),
            "cameras": cameras,
            "resolution": meta.get("resolution"),
            "duration_s": meta.get("duration_s"),
            "sha256": {
                META_FILE: sha256(corpus / META_FILE),
                MANIFEST_FILE: sha256(corpus / MANIFEST_FILE),
                FRAMES_FILE: sha256(zip_path),
            },
        },
        "selection": {
            "method": "evenly_spaced_frame_sets_integer_half_up",
            "formula": "floor((i*(N-1)+floor((K-1)/2))/(K-1)), i=0..K-1",
            "reason": "cover the complete recording and every camera deterministically",
        },
        "calibration": {
            "frame_set_count": len(selected_ids),
            "image_count": len(calibration_images),
            "first_timestamp_s": min(timestamps),
            "last_timestamp_s": max(timestamps),
            "frame_set_ids": selected_ids,
            "images": calibration_images,
        },
        "validation_holdout": {
            "frame_set_count": len(validation_ids),
            "image_count": len(validation_ids) * len(cameras),
            "frame_set_ids": validation_ids,
        },
        "notes": [
            "Calibration and holdout IDs are disjoint and their union is the full corpus.",
            "The holdout is for quantization engineering, not a final generalization claim.",
            "The recording is hover-only and temporally correlated; collect a separate flight later.",
        ],
    }


def validate_split(split):
    all_ids = set(range(split["corpus"]["frame_sets"]))
    calibration = split["calibration"]
    holdout = split["validation_holdout"]
    calibration_ids = set(calibration["frame_set_ids"])
    holdout_ids = set(holdout["frame_set_ids"])
    if calibration_ids & holdout_ids:
        raise ValueError("calibration and validation holdout overlap")
    if calibration_ids | holdout_ids != all_ids:
        raise ValueError("calibration and validation holdout do not cover the corpus")
    if len(calibration_ids) != calibration["frame_set_count"]:
        raise ValueError("calibration frame-set count is inconsistent")
    if len(holdout_ids) != holdout["frame_set_count"]:
        raise ValueError("validation holdout frame-set count is inconsistent")
    expected_images = len(calibration_ids) * len(split["corpus"]["cameras"])
    if calibration["image_count"] != expected_images:
        raise ValueError("calibration image count is inconsistent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--calibration-sets", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    output = Path(args.output) if args.output else corpus / DEFAULT_OUTPUT
    split = build_split(corpus, calibration_sets=args.calibration_sets)
    validate_split(split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    calibration = split["calibration"]
    holdout = split["validation_holdout"]
    print(f"wrote {output}")
    print(f"calibration: {calibration['frame_set_count']} frame sets, "
          f"{calibration['image_count']} images")
    print(f"holdout:     {holdout['frame_set_count']} frame sets, "
          f"{holdout['image_count']} images")
    print(f"coverage:    {calibration['first_timestamp_s']:.3f} to "
          f"{calibration['last_timestamp_s']:.3f} seconds")


if __name__ == "__main__":
    main()
