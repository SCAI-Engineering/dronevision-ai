# Running DroneVision Benchmarks on the NXP i.MX93 FRDM Board

This document provides step-by-step instructions for deploying, running, and pulling benchmark results for DroneVision on the NXP i.MX93 FRDM board (Arm Cortex-A55 CPU + Arm Ethos-U65 NPU).

---

## 1. Prerequisites & Environment Setup

### Host Machine Requirements
- SSH access configured in `~/.ssh/config` for host alias `imx`.
- Python 3.12 environment (`.venv-export`) with pinned exporter tools.

### i.MX93 Board Environment
- Linux kernel with `/dev/ethosu0` character device enabled.
- NXP Ethos-U delegate library at `/usr/lib/libethosu_delegate.so`.
- Python 3.12 with NXP system site packages (`numpy`, `opencv-python`, `onnxruntime`, `tflite_runtime`/`ai_edge_litert`).

---

## 2. One-Command Board Setup & Sync

From the host machine repository root (`dronevision-ai`):

```bash
# 1. Verify board connectivity and CPU/NPU features
HOST=imx ./bench/pi.sh info

# 2. Sync committed code tree to the i.MX93 board (~/dronevision-ai)
HOST=imx BRANCH=feature/pi5-bench ./bench/pi.sh sync

# 3. Transfer compiled Vela INT8 NPU model artifacts to the board
tar -c bench/out/vela | ssh imx "mkdir -p ~/dronevision-ai && tar -x -C ~/dronevision-ai"

# 4. Initialize board virtual environment (first time only)
HOST=imx BRANCH=feature/pi5-bench SYSTEM_SITE_PACKAGES=1 BOARD_TAG=imx93 ./bench/pi.sh setup
```

---

## 3. Executing Benchmarks on i.MX93

### A. Isolated NPU Inference Speed Test
Test single-image NPU execution, verify `/dev/ethosu0` delegation, and measure pre-processing, NPU inference, and post-processing latency:

```bash
HOST=imx ./bench/pi.sh run -m bench.test_imx_npu
```

### B. Full 337-Corpus 3D Accuracy & Latency (Sequential NPU Execution)
Run complete 3D localization across all 337 synchronized 4-camera frame sets on the Ethos-U65 NPU:

```bash
HOST=imx ./bench/pi.sh run -m bench.accuracy \
  --detector yolo \
  --runtime tflite \
  --model bench/out/vela/drone_yolo26n_v4_raw_int8_vela.tflite
```

### C. Full 337-Corpus 3D Accuracy & Latency (2-Worker Pipelined NPU)
Overlap frame decoding and pre-processing across 2 camera workers with Ethos-U NPU hardware execution:

```bash
HOST=imx ./bench/pi.sh run -m bench.accuracy \
  --detector yolo \
  --runtime tflite \
  --model bench/out/vela/drone_yolo26n_v4_raw_int8_vela.tflite \
  --parallel 2
```

### D. FP32 CPU Baseline Benchmark (For Comparison)
Measure the 2-core Cortex-A55 FP32 ONNX CPU baseline to calculate hardware speedup:

```bash
HOST=imx ./bench/pi.sh run -m bench.accuracy \
  --detector yolo \
  --runtime onnx \
  --threads 2
```

---

## 4. Pulling Results & Comparative Reporting

Pull generated benchmark JSON files back to the host machine and render the comparison against Raspberry Pi 4 and Raspberry Pi 5:

```bash
# Pull result files into bench/out/
HOST=imx ./bench/pi.sh pull

# Generate comparative Markdown table (i.MX93 CPU vs NPU vs Pi 4 / Pi 5)
.venv-export/bin/python -m bench.report_imx_vs_pi
```
