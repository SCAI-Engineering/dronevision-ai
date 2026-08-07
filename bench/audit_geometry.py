#!/usr/bin/env python3
"""Audit of Projection Matrices via Reprojection Error analysis.

This tool measures the "White Box" accuracy of the system. Instead of measuring 
the final 3D position error (a symptom), it measures the reprojection error 
(the cause). 

It asks: "Given that the drone is exactly at the ground-truth 3D position, 
how many pixels away is the actual detection?"

A high reprojection error for a specific camera indicates that its Projection 
Matrix (P) is inaccurate, leading to 'drift' in the final triangulation.

    python -m bench.audit_geometry --detector color
    python -m bench.audit_geometry --detector yolo --runtime onnx
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dronevision.l4_triangulation.calibration import load_site
from dronevision.l4_triangulation.geometry import reproj_err
from dronevision.l2_perception.detector import make_detector
from dronevision.io.sources.replay import ReplaySource

def stats_px(values):
    a = np.asarray(list(values), float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size), 
        "mean": a.mean(), 
        "median": np.median(a),
        "p95": np.percentile(a, 95), 
        "max": a.max(),
        "std": a.std()
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--detector", default="color")
    ap.add_argument("--runtime", default=None, help="yolo only: ultralytics | onnx | executorch")
    ap.add_argument("--imgsz", type=int, default=320)
    a = ap.parse_args(argv)

    site = load_site(a.site)
    
    # Initialize detector to get the actual detections from images
    det_kw = {}
    if a.detector in ("yolo", "hybrid"):
        det_kw = {"runtime": a.runtime, "imgsz": a.imgsz}
    detector = make_detector(a.detector, **det_kw)

    # Load corpus
    src = ReplaySource(a.corpus)
    print(f"Auditing geometry for site: {site.name}")
    print(f"Using detector: {detector.name}")
    print(f"Loading corpus from: {a.corpus}...")

    # Store errors per camera: {cam_name: [errors]}
    cam_errors = {name: [] for name in site.cam_names}
    misses = 0
    total_frames = 0

    for s in src:
        total_frames += 1
        truth = s.truth
        if truth is None:
            continue

        # For each camera, get the detection and calculate reprojection error against truth
        for cam in site.cam_names:
            img = s.latest(cam)
            if img is None:
                continue
            
            uv_detected = detector.detect(img, cam=cam)
            if uv_detected:
                # Use the geometry primitive to find pixel distance between 
                # projected truth and actual detection
                err = reproj_err(cam, truth, uv_detected, cal=site)
                cam_errors[cam].append(err)
            else:
                misses += 1

    print("\n--- Reprojection Error Audit (Pixels) ---")
    print(f"Total frames processed: {total_frames}")
    print(f"Detections missed: {misses}")
    print()
    print(f"{'Camera':<12} {'N':>6} {'Mean':>8} {'Median':>8} {'p95':>8} {'Max':>8} {'Std':>8}")
    print("-" * 60)

    all_stats = []
    for cam in site.cam_names:
        s = stats_px(cam_errors[cam])
        if s:
            print(f"{cam:<12} {s['n']:>6} {s['mean']:>8.2f} {s['median']:>8.2f} {s['p95']:>8.2f} {s['max']:>8.2f} {s['std']:>8.2f}")
            all_stats.append(s['mean'])
        else:
            print(f"{cam:<12} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    if all_stats:
        avg_mean = np.mean(all_stats)
        print("-" * 60)
        print(f"{'AVERAGE':<12} {'-':>6} {avg_mean:>8.2f} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
        print()

        # Diagnosis logic
        max_err_cam = max(cam_errors, key=lambda k: np.mean(cam_errors[k]) if cam_errors[k] else 0)
        max_val = np.mean(cam_errors[max_err_cam]) if cam_errors[max_err_cam] else 0
        
        if max_val > 5.0: # Threshold for "significant" drift in pixels
            print(f"[DIAGNOSIS]: Significant reprojection error detected in {max_err_cam} ({max_val:.2f} px).")
            print("This suggests the Projection Matrix (P) for this camera is inaccurate.")
            print("Consider performing Bundle Adjustment to refine the calibration.")
        else:
            print("[DIAGNOSIS]: Reprojection errors are within acceptable limits (< 5px).")
            print("Geometry is mathematically sound; any position error is likely due to detector noise.")

    src.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
