#!/usr/bin/env python3
"""Evaluate Step 3.5 quantization-only correctness gate on TFLite CPU.

Evaluates accuracy across:
1. 64 calibration frame sets
2. 273 holdout frame sets
3. All 337 frame sets

Usage:
    python -m bench.eval_quant_gate
"""
import json
import time
from pathlib import Path

import numpy as np

from dronevision.io.sources.replay import ReplaySource
from dronevision.l2_perception.detector import YoloDetector
from dronevision.l4_triangulation.calibration import load_site
from dronevision.l5_estimation.smoothing import EMASmoother
from dronevision.pipeline import LocalizationPipeline


SPLIT_MANIFEST = "data/corpus/quantization_split.json"


def stats_mm(values):
    a = np.asarray(list(values), float) * 1000
    if a.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def run_eval_subset(pipeline, corpus_path, target_ids):
    err3, errxy, errz = [], [], []
    misses = 0
    target_set = set(target_ids)
    source = ReplaySource(corpus_path, mode="decoded")

    for _ in source:
        rec = source._records[source._i]
        frame_id = rec["i"]
        if frame_id not in target_set:
            continue

        truth = source.truth
        est = pipeline.locate_from(source)

        if est is None or truth is None:
            misses += 1
            continue

        d = np.array(est.position) - truth
        err3.append(float(np.linalg.norm(d)))
        errxy.append(float(np.linalg.norm(d[:2])))
        errz.append(float(d[2]))

    source.close()
    s3 = stats_mm(err3)
    sxy = stats_mm(errxy)
    sz = stats_mm(np.abs(errz))

    return {
        "requested": len(target_ids),
        "localized": s3["n"],
        "misses": misses,
        "3d_error_mm": s3,
        "horiz_error_mm": sxy,
        "vert_error_mm": sz,
    }


def main():
    split_path = Path(SPLIT_MANIFEST)
    if not split_path.is_file():
        raise SystemExit(f"missing split manifest: {split_path}")

    with open(split_path, "r", encoding="utf-8") as fh:
        split_data = json.load(fh)

    calib_ids = split_data["calibration"]["frame_set_ids"]
    holdout_ids = split_data["validation_holdout"]["frame_set_ids"]
    all_ids = calib_ids + holdout_ids

    site = load_site("factory")
    corpus_path = "data/corpus"

    models_to_test = [
        ("FP32 ONNX (Raw)", "onnx", "drone_yolo26n_v4_raw.onnx"),
        ("FP32 TFLite (Raw)", "tflite", "drone_yolo26n_v4_raw_fp32.tflite"),
        ("INT8 TFLite (Raw)", "tflite", "drone_yolo26n_v4_raw_int8.tflite"),
    ]

    report = {}

    for label, runtime_name, model_name in models_to_test:
        print(f"=== Evaluating {label} ({model_name}) ===")
        detector = YoloDetector(model=model_name, runtime=runtime_name, imgsz=320)
        pipe = LocalizationPipeline(cal=site, detector=detector, smoother=EMASmoother(0.0), timing=True)

        calib_res = run_eval_subset(pipe, corpus_path, calib_ids)
        holdout_res = run_eval_subset(pipe, corpus_path, holdout_ids)
        full_res = run_eval_subset(pipe, corpus_path, all_ids)

        report[model_name] = {
            "label": label,
            "runtime": runtime_name,
            "calibration_64": calib_res,
            "holdout_273": holdout_res,
            "full_337": full_res,
        }

        print(f"  Holdout (273): localized={holdout_res['localized']}/{holdout_res['requested']}, "
              f"3D mean={holdout_res['3d_error_mm']['mean']:.2f} mm, P95={holdout_res['3d_error_mm']['p95']:.2f} mm")
        print(f"  Full (337):    localized={full_res['localized']}/{full_res['requested']}, "
              f"3D mean={full_res['3d_error_mm']['mean']:.2f} mm, P95={full_res['3d_error_mm']['p95']:.2f} mm\n")

    out_file = Path("data/scratch/quant_gate_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote quantization gate evaluation report to {out_file}")


if __name__ == "__main__":
    main()
