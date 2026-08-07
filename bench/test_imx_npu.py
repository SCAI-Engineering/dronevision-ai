#!/usr/bin/env python3
"""Smoke test and benchmark for Ethos-U NPU delegation on i.MX93 board.

Usage:
    python -m bench.test_imx_npu
"""
import time
import numpy as np

from dronevision.io.sources.replay import ReplaySource
from dronevision.l2_perception.runtimes.tflite_rt import TfLiteRuntime


MODEL_PATH = "bench/out/vela/drone_yolo26n_v4_raw_int8_vela.tflite"
DELEGATE_PATH = "/usr/lib/libethosu_delegate.so"


def main():
    print("=== Testing Arm Ethos-U65 NPU Delegation on i.MX93 ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Delegate: {DELEGATE_PATH}")

    rt = TfLiteRuntime(MODEL_PATH, delegate_path=DELEGATE_PATH)
    print(f"Model described: {rt.describe()}")

    src = ReplaySource("data/corpus", mode="decoded")
    frame = next(iter(src)).latest("cam_ne")

    print("\nWarmup passes (10 calls)...")
    for _ in range(10):
        rt.detect_boxes(frame)

    print("\nTimed passes (50 calls)...")
    t0 = time.perf_counter()
    for _ in range(50):
        dets = rt.detect_boxes(frame)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    means = rt.stats.means()

    print(f"\n--- NPU Execution Results ---")
    print(f"50 inferences total wall time: {elapsed:.3f} s ({elapsed/50*1000:.2f} ms/frame)")
    print(f"Per-stage timings:")
    print(f"  Pre-processing:  {means['pre_ms']:.3f} ms")
    print(f"  NPU Inference:   {means['infer_ms']:.3f} ms")
    print(f"  Post-processing: {means['post_ms']:.3f} ms")
    print(f"  Total per frame: {means['total_ms']:.3f} ms")
    print(f"  Compute ceiling: {1000.0/means['total_ms']:.2f} FPS")


if __name__ == "__main__":
    main()
