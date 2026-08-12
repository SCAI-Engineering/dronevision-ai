# DroneVision on NXP i.MX93 FRDM

Validation record for running DroneVision on the NXP i.MX93 FRDM board available through
the `imx` SSH host. The measurements in this document were taken on 6 August 2026.

## Executive summary

DroneVision installs and runs correctly on the board with the NXP-provided Python, OpenCV
and ONNX Runtime packages. The complete test suite passes and the bundled FP32 ONNX model
produces valid output on the CPU.

The classical colour-marker pipeline processes approximately 33 four-camera frame sets
per second. The YOLO FP32 CPU path is far slower: with both Cortex-A55 cores it processes
approximately 1.02 frame sets per second, or 4.09 individual camera inferences per second.
This makes learned detection—not JPEG decoding or geometry—the dominant optimization
opportunity.

The next meaningful optimization is an INT8 Ethos-U65 backend. JPEG decoding and the
geometric stages are not significant bottlenecks.

## Provenance

| Item | Value |
|---|---|
| Source branch | `feature/pi5-bench` |
| Source commit tested | `e9b6088bea8ab801b2a1b3d6f852487d87d47543` |
| Commit subject | `Measure multiprocessing against threads: 12% slower on the Pi 4` |
| Remote deployment | `/root/dronevision-ai` |
| Python environment | `/root/dronevision-ai/.venv` |
| Corpus samples | 337 four-camera frame sets |
| Corpus content | 30 seconds, four JPEG cameras, synchronized ground truth |

Artifact identity:

| Artifact | Size | SHA-256 |
|---|---:|---|
| `models/drone_yolo26n_v4.onnx` | 9,678,386 bytes | `7acd721e718a6fe444b22eb93300d2316ae4413209c2a89789072d4e1a4a3bc9` |
| `data/corpus/frames.zip` | 12,864,307 bytes | `21596ae0161babdd593170378915a4d5b51da9d3adcb96a59b0998dd2e11d885` |

## Board inventory

### Hardware and operating system

| Resource | Observed value |
|---|---|
| Board | NXP i.MX93 11x11 FRDM |
| Application CPU | 2x Arm Cortex-A55 |
| Architecture | AArch64 |
| CPU features of interest | NEON, FP16 and Arm dot-product (`asimddp`) |
| Real-time core | Cortex-M33 available in the platform |
| RAM | 1,879 MiB usable, no swap |
| NPU | Arm Ethos-U65 exposed as `/dev/ethosu0` |
| NPU reserved memory | 128 MiB |
| OS | NXP i.MX Release Distro 6.6 Scarthgap |
| Kernel | `6.6.36-lts-next-g34fd186d1571` |
| Root filesystem | 6.7 GiB ext4, approximately 2.5 GiB free during testing |
| Physical eMMC | 28.9 GiB / nominal 32 GB |

The eMMC currently has an 83 MiB boot partition and a 6.9 GiB root partition. Roughly
22 GiB are unpartitioned. Expanding the root partition would be useful before storing
larger datasets or multiple model variants.

### Connected and exposed devices

- HDMI is connected, with 1280x720 and 800x600 modes.
- Weston/Wayland and a terminal are active.
- A Logitech USB keyboard/mouse receiver is connected.
- Three I2C buses, SPI, GPIO, CAN, OP-TEE and the Ethos-U device are exposed.
- `can0` exists but is down.
- No `/dev/video*` camera device is currently enumerated.
- Audio playback is available through `mqs-audio`; no capture device was enumerated.

### Network state

`eth0` negotiated at 100 Mbps during inspection and had two IPv4 addresses,
`192.168.1.128` and `192.168.1.132`. Both ConnMan and systemd-networkd were managing the
interface and requesting DHCP. This should be resolved before live latency measurements.

The system exposes SSH, RPCBind, NTP and Avahi. SSH permits direct root login and password
authentication, while the firewall policies are `ACCEPT`. This is acceptable only on an
isolated development network and should be hardened for deployment.

### Installed development and AI stack

| Component | Version / state |
|---|---|
| Python | 3.12.4 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0 NXP build |
| PyYAML | 6.0.1 |
| ONNX Runtime | 1.17.1 |
| ONNX providers | `CPUExecutionProvider` only |
| TensorFlow Lite runtime | 2.16.2 |
| NNStreamer | 2.2.0 |
| Ethos-U driver stack and firmware | 24.05 |
| Vela compiler | 3.12.0 |
| GStreamer | 1.24.0 NXP build |
| GCC | 13.3.0 |
| CMake | 3.28.3 |
| Weston | 12.0.4 |

Docker and Podman are not installed. The full Gazebo/PX4 simulator is neither required nor
appropriate for these board-side offline benchmarks.

## Phase 0: preparation and traceability

Status: complete.

The committed source tree was transferred over SSH without requiring Git credentials on
the board. A virtual environment was created with `--system-site-packages`, and
DroneVision was installed editable with `--no-deps`. This deliberately preserves the NXP
builds of NumPy, OpenCV and ONNX Runtime instead of replacing them with generic wheels.

The effective module paths were verified as:

```text
numpy        /usr/lib/python3.12/site-packages/numpy/__init__.py
opencv       /usr/lib/python3.12/site-packages/cv2/__init__.py
onnxruntime  /usr/lib/python3.12/site-packages/onnxruntime/__init__.py
dronevision  /root/dronevision-ai/dronevision/__init__.py
```

`bench/pi.sh` was generalized for Arm SBCs:

- transfers are gzip-compressed;
- `THREADS=auto` tests one thread and the remote core count;
- `SYSTEM_SITE_PACKAGES=1` preserves vendor-optimized packages;
- `BOARD_TAG` controls target-specific output names;
- the script is executable.

The setup command for this board is:

```bash
HOST=imx \
BRANCH=feature/pi5-bench \
SYSTEM_SITE_PACKAGES=1 \
BOARD_TAG=imx93 \
./bench/pi.sh setup
```

## Phase 1: functional smoke test

Status: complete.

### Corpus and configuration

`factory.yaml` loaded correctly. The first corpus set contained all four expected cameras:

```text
cam_ne  360x640x3 RGB
cam_nw  360x640x3 RGB
cam_sw  360x640x3 RGB
cam_se  360x640x3 RGB
truth   [0.00961, 0.08076, 2.45177]
```

### ONNX smoke inference

One direct `OnnxRuntime` inference was run on `cam_ne`, without a benchmark sweep or
automatic warm-up passes.

| Property | Result |
|---|---|
| Provider | `CPUExecutionProvider` |
| Precision | FP32 |
| Input | `images[1, 3, 320, 320]` |
| Output | `output0[1, 300, 6]` |
| Output layout | YOLO end-to-end |
| ONNX intra-op threads | 2 |
| Session creation | 703.952 ms |
| Preprocess | 10.262 ms |
| First inference | 252.477 ms |
| Postprocess | 0.533 ms |
| Maximum RSS | 131,648 KiB |
| Temperature | 57.35 to 58.35 degrees C |

The selected first `cam_ne` image produced no detection above confidence 0.25. This was a
valid empty output, not a runtime error; the complete-corpus tests below measured the real
detection and localization rate.

### Test suite

`pytest 9.1.1` was installed only inside the project virtual environment.

```text
184 passed, 3 skipped, 0 failed in 31.38s
```

The skipped tests cover optional components. No core test failed.

## Phase 2: accuracy and end-to-end timing

Status: complete.

The tests were run serially to avoid CPU, memory and thermal interference:

```bash
# Classical marker baseline
.venv/bin/python -m bench.accuracy

# YOLO CPU baseline
.venv/bin/python -m bench.accuracy \
  --detector yolo --runtime onnx --threads 1

# Use both Cortex-A55 cores
.venv/bin/python -m bench.accuracy \
  --detector yolo --runtime onnx --threads 2

# Include JPEG reading and decode
.venv/bin/python -m bench.accuracy \
  --detector yolo --runtime onnx --threads 2 --mode encoded
```

| Configuration | Localized | Misses | Mean 3D error | P95 3D error | Mean total | Frame sets/s |
|---|---:|---:|---:|---:|---:|---:|
| Colour, decoded | 337 | 0 | 6.22 mm | 10.72 mm | 30.225 ms | 33.09 |
| YOLO ONNX FP32, 1 thread | 333 | 4 | 36.42 mm | 77.58 mm | 1,607.158 ms | 0.622 |
| YOLO ONNX FP32, 2 threads | 333 | 4 | 36.42 mm | 77.58 mm | 977.734 ms | 1.023 |
| YOLO ONNX FP32, 2 threads, JPEG | 333 | 4 | 36.42 mm | 77.58 mm | 990.761 ms | 1.009 |
| **YOLO INT8 Ethos-U65 NPU (Sequential)** | **336** | **1** | **52.39 mm** | **95.94 mm** | **199.369 ms** | **5.02** |
| **YOLO INT8 Ethos-U65 NPU (2 Workers)** | **336** | **1** | **52.39 mm** | **95.94 mm** | **167.777 ms** | **5.96** |

YOLO localized 98.81% of the corpus frame sets on FP32 and 99.70% (336/337) on Ethos-U NPU. Precision was
identical across thread counts and input modes, as expected.

### Timing breakdown

| Configuration | Acquire | Detect | Triangulate | Total | Wall time |
|---|---:|---:|---:|---:|---:|
| Colour, decoded | 0.060 ms | 25.841 ms | 4.237 ms | 30.225 ms | 10.31 s |
| YOLO, 1 thread, decoded | 0.060 ms | 1,604.101 ms | 2.918 ms | 1,607.158 ms | 541.71 s |
| YOLO, 2 threads, decoded | 0.060 ms | 974.626 ms | 2.968 ms | 977.734 ms | 329.59 s |
| YOLO, 2 threads, JPEG | 14.385 ms | 973.363 ms | 2.936 ms | 990.761 ms | 333.99 s |
| **YOLO Ethos-U NPU (Sequential)** | **0.059 ms** | **195.899 ms** | **3.330 ms** | **199.369 ms** | **67.30 s** |
| **YOLO Ethos-U NPU (2 Workers)** | **0.060 ms** | **164.176 ms** | **3.459 ms** | **167.777 ms** | **56.66 s** |

Two threads provide a 1.644x speed-up and reduce total latency by 39.16% relative to one
thread. The process consumed approximately 173-175% CPU, showing that ONNX used both cores
but did not scale perfectly because of serial work and synchronization.

JPEG adds approximately 13.0 ms per four-camera set, only 1.33% of total latency. The
dominant cost is therefore inference, not acquisition, decode, association or geometry.

### Memory and thermal observations

- Decoded mode retained much of the expanded corpus and reached approximately 54% process
  memory, leaving about 676 MiB available system-wide.
- Encoded mode used approximately 8% process memory and left about 1.55 GiB available.
- Encoded mode is strongly preferred on this 2 GB board.
- The two-thread runs stabilized around 59-60 degrees C.
- No throttling, OOM, kernel error or failed inference was observed.
- After testing, the board returned to approximately 53 degrees C and 1.6 GiB available.

## Workload assessment

Observed two-core CPU/YOLO result:

```text
4.09 estimated camera inferences/s
1.02 four-camera frame sets/s
977.7 ms per decoded frame set
```

The FP32 CPU implementation is almost an order of magnitude slower than the later NPU
implementation and makes learned perception the clear system bottleneck.

The colour detector achieves 33 frame sets/s, but it depends on a visible colour marker
and does not replace general drone detection.

## Comparison context: Raspberry Pi 5

The i.MX93 has two Cortex-A55 cores and 2 GB RAM, while a Raspberry Pi 5 has four newer
Cortex-A76 cores and is available with substantially more RAM. The Pi 5 is expected to be
much faster for FP32 ONNX CPU inference and general development workloads.

The i.MX93 advantages are different: integrated 0.5-TOPS Ethos-U65, Cortex-M33,
CAN-FD, eMMC, EdgeLock and embedded/industrial I/O. The correct comparison for this board
is therefore not CPU FP32 alone, but optimized INT8 inference on its integrated NPU.

## Limitations of these results

- ONNX Runtime exposes only `CPUExecutionProvider`; the Ethos-U65 was not used.
- The bundled ONNX model is FP32, not INT8.
- The corpus is hover-only, with little horizontal movement. It does not fully exercise
  tracking, coasting or motion-gated detection.
- Weston and the normal NXP development services remained active. These are results for
  the board as found, not a stripped headless image.
- The live ZeroMQ path was not tested. `pyzmq` is not currently installed on the board.
- No physical camera is enumerated, so all completed measurements used the bundled corpus.
- Power draw was not instrumented.

## Remaining test plan

### Phase 3: isolated speed benchmark

Run repeated, warmed inference measurements with one and two threads, collecting mean,
median, P95 and thermal context through `bench.speed`. This separates model execution from
the full localization loop and produces directly comparable JSON rows.

### Phase 4: multicamera scheduling

Compare sequential execution, two camera workers and oversubscription. Test detect-every-N
plus tracking if every-frame inference cannot meet the budget.

### Phase 5: sustained thermal test

Run the best CPU case for 10-15 minutes, first with the system as found and then with
Weston temporarily stopped. Record CPU temperature, frequency, memory and latency drift.

### Phase 6: live network test

Resolve duplicate network management, install `pyzmq` in the virtual environment, and
measure camera JPEG transport, decode, inference, stale frames and recovery from one or
two missing cameras.

### Phase 7: Ethos-U65 path

1. Export or convert a TensorFlow Lite INT8 model.
2. Quantize with representative images from the corpus.
3. Check operator compatibility.
4. Compile with Vela for Ethos-U65.
5. Measure the delegated portion of the graph.
6. Add a TFLite/Ethos-U runtime to DroneVision.
7. Repeat accuracy, speed, multicamera and thermal tests.

Success requires accuracy close to the FP32 reference and a substantial reduction in the
current 11.7x real-time performance gap.
