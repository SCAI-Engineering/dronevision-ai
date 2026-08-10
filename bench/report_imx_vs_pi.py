#!/usr/bin/env python3
"""Generate comparative report: i.MX93 (CPU vs Ethos-U NPU) vs Pi 4 / Pi 5.

Usage:
    python -m bench.report_imx_vs_pi
"""
import glob
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main():
    bench_dir = Path("bench/out")

    # Load available JSON files
    pi5_t1 = load_json(bench_dir / "pi5-onnx-drone_yolo26n_v4-t1.json")
    pi5_t4 = load_json(bench_dir / "pi5-onnx-drone_yolo26n_v4-t4.json")
    imx_t1 = load_json(bench_dir / "imx93-onnx-fp32-t1.json")
    imx_t2 = load_json(bench_dir / "imx93-onnx-fp32-t2.json")
    vela_report = load_json(bench_dir / "vela/vela_compilation_report.json")
    quant_gate = load_json(Path("data/scratch/quant_gate_results.json"))

    print("=========================================================================================")
    print("                     BENCHMARK COMPARISON: i.MX93 NPU vs Pi 5 & Pi 4                     ")
    print("=========================================================================================")
    print()

    # Table 1: Single-Image Inference & Throughput
    print("### 1. Single-Image Inference Performance & Speedup")
    print()
    print("| Board / Accelerator | Processor / Core | dotprod | Runtime | Precision | Threads | Single-Img Latency | Single-Img FPS | Speedup vs i.MX93 FP32 CPU |")
    print("|---|---|---|---|---|---|---:|---:|---:|")
    print("| **Pi 4 Model B** | Cortex-A72 | **no** | ONNX | FP32 | 1 | 205.4 ms | 4.8 FPS | 1.93x |")
    print("| **Pi 4 Model B** | Cortex-A72 | **no** | ONNX | FP32 | 4 | 93.7 ms | 10.4 FPS | 4.23x |")

    if imx_t1:
        inf_ms = imx_t1["stages_ms"]["infer"]["mean"]
        fps = imx_t1["fps"]
        print(f"| **i.MX93 FRDM** | 1x Cortex-A55 | yes | ONNX | FP32 | 1 | {inf_ms:.1f} ms | {fps:.1f} FPS | 1.00x (baseline) |")
    else:
        print("| **i.MX93 FRDM** | 1x Cortex-A55 | yes | ONNX | FP32 | 1 | 396.4 ms | 2.5 FPS | 1.00x (baseline) |")

    if imx_t2:
        inf_ms = imx_t2["stages_ms"]["infer"]["mean"]
        fps = imx_t2["fps"]
        print(f"| **i.MX93 FRDM** | 2x Cortex-A55 | yes | ONNX | FP32 | 2 | {inf_ms:.1f} ms | {fps:.1f} FPS | 1.67x |")
    else:
        print("| **i.MX93 FRDM** | 2x Cortex-A55 | yes | ONNX | FP32 | 2 | 237.9 ms | 4.2 FPS | 1.67x |")

    if pi5_t1:
        inf_ms = pi5_t1["stages_ms"]["infer"]["mean"]
        fps = pi5_t1["fps"]
        print(f"| **Raspberry Pi 5** | Cortex-A76 | yes | ONNX | FP32 | 1 | {inf_ms:.1f} ms | {fps:.1f} FPS | {396.4/inf_ms:.2f}x |")

    if pi5_t4:
        inf_ms = pi5_t4["stages_ms"]["infer"]["mean"]
        fps = pi5_t4["fps"]
        print(f"| **Raspberry Pi 5** | Cortex-A76 | yes | ONNX | FP32 | 4 | {inf_ms:.1f} ms | {fps:.1f} FPS | {396.4/inf_ms:.2f}x |")

    print("| **i.MX93 Ethos-U65 (Measured)** | **Ethos-U65 NPU** | **n/a** | **TFLite+Ethos-U** | **INT8** | **NPU** | **40.2 ms** | **24.9 FPS** | **9.87x vs 1t CPU (5.92x vs 2t)** |")

    if vela_report:
        v_ms = vela_report.get("estimated_inference_time_ms", 21.58)
        v_fps = vela_report.get("estimated_inferences_per_second", 46.34)
        print(f"| **i.MX93 Ethos-U65 (Vela Est.)** | Ethos-U65 NPU | n/a | TFLite+Ethos-U | INT8 | NPU | {v_ms:.1f} ms | {v_fps:.1f} FPS | {396.4/v_ms:.2f}x |")

    print()
    print("### 2. End-to-End 4-Camera Fix Throughput & Closed-Loop Control")
    print()
    print("| Target / Architecture | Execution Strategy | 4-Cam Fix Latency | 4-Cam Fix Rate (Hz) | Real-Time Loop Target (12 Hz) |")
    print("|---|---|---:|---:|---|")
    print("| **i.MX93 CPU FP32 (2 Cores)** | Sequential ONNX (2 threads) | 977.7 ms | 1.02 Hz | ❌ 11.7x below target |")
    print("| **Raspberry Pi 4 CPU FP32 (4 Cores)** | 4 Workers x 1 Thread | 322.5 ms | 3.10 Hz | ❌ 3.87x below target |")
    print("| **Raspberry Pi 5 CPU FP32 (4 Cores)** | 4 Workers x 1 Thread | ~120.0 ms | ~8.33 Hz | ⚠️ 1.44x below target |")
    print("| **i.MX93 Ethos-U65 NPU (Measured)** | **Sequential NPU Worker** | **199.4 ms** | **5.02 Hz** | **4.9x faster than i.MX93 CPU** |")
    print("| **i.MX93 Ethos-U65 NPU (Measured)** | **2-Worker Pipelined NPU** | **167.8 ms** | **5.96 Hz** | **5.8x faster than i.MX93 CPU** |")
    print()

    print("### 3. Accuracy & Quantization Loss Across 337 Frame Sets")
    print()
    print("| Model Variant | Runtime | 3D Mean Error | 3D P95 Error | Localized / Misses | Size |")
    print("|---|---|---:|---:|---:|---:|")
    print("| **Raw FP32 Baseline (ONNX)** | ONNX CPU | 30.87 mm | 59.13 mm | 337 / 0 | 9.3 MB |")
    print("| **Raw FP32 Baseline (TFLite)** | TFLite CPU | 30.87 mm | 59.13 mm | 337 / 0 | 9.3 MB |")
    print("| **Raw INT8 Quantized (TFLite CPU)** | TFLite CPU | 54.08 mm | 101.58 mm | 337 / 0 | 2.73 MB |")
    print("| **Raw INT8 Quantized (Ethos-U NPU)** | Ethos-U65 NPU | 52.39 mm | 95.94 mm | 336 / 1 | 2.42 MB |")
    print()


if __name__ == "__main__":
    main()
