# INT8 quantization and Ethos-U65 plan for the i.MX93

This document preserves the current state and the intended next steps for quantizing the
DroneVision detector and running it on the Arm Ethos-U65 in the NXP i.MX93 FRDM board.
It was last updated on 9 August 2026. Commands below are plans or command templates unless
their section explicitly says that they have already been run.

## Current checkpoint

| Work item | Status | Evidence / artifact |
|---|---|---|
| FP32 i.MX93 CPU baseline and Pi 5 comparison | Complete | `docs/IMX93_VALIDATION.md` |
| Ethos-U65 platform sanity check | Complete | MobileNet V1, 100% NPU placement, 3.78 ms |
| Phase 1: calibration/holdout split | Complete and committed | commit `df30963`, `data/corpus/quantization_split.json` |
| Phase 2: raw FP32 YOLO export | Complete | `models/drone_yolo26n_v4_raw.onnx` |
| Phase 2: full-corpus raw FP32 validation | Complete | 337/337, 30.87 mm mean, 59.13 mm P95 |
| Phase 3: full-integer TFLite conversion | Complete | `models/drone_yolo26n_v4_raw_int8_io.tflite`: INT8 `[1,3,320,320]` input and INT8 `[1,5,2100]` output |
| Phase 4: Vela compatibility and compilation | Complete | `bench/out/vela/drone_yolo26n_v4_raw_int8_io_vela.tflite`: 1/1 node delegated, no CPU boundary nodes |
| Phase 5: DroneVision TFLite/Ethos-U runtime | Complete | `dronevision/l2_perception/runtimes/tflite_rt.py` (`/usr/lib/libethosu_delegate.so` backend) |
| Phase 6: NPU accuracy, speed and scheduling | Complete | 40.18 ms isolated inference, 167.8 ms (5.96 Hz) 4-cam 2-worker fix, 52.39 mm 3D error |
| Separate CPU track: static INT8 ONNX | Pending | Does not block the NPU track |

Repository state at this checkpoint:

- `df30963` is the committed phase-1/NPU-context checkpoint.
- Phase-2 source, documentation and model changes are present in the working tree but have
  not yet been committed.
- The final integer-I/O artifact has been executed on the NPU and validated on the full
  337-set corpus.

## Why this work is necessary

The current detector is a FP32 ONNX model executed by ONNX Runtime on the two Cortex-A55
cores. The best measured CPU scheduling strategy is two camera workers with one ONNX
thread each:

| Measurement | Current i.MX93 result |
|---|---:|
| Isolated inference, one CPU thread | 396.4 ms/image |
| Isolated inference, two CPU threads | 237.9 ms/image |
| Best four-camera fix | 875-882 ms/fix |
| Best sustained rate | 1.14 fixes/s |

The original end-to-end FP32 reference is 36.42 mm mean 3D error, with 333 localized
samples and 4 misses out of 337. The phase-2 raw export improves this to 30.87 mm and
337/337 localized; it is now the primary reference for quantization. The Pi 5 is
approximately 7-8x faster than the i.MX93 for the complete FP32 CPU pipeline, so CPU
optimization alone is unlikely to make the i.MX93 real-time. The board's differentiating
resource is its integrated 0.5-TOPS Ethos-U65 NPU.

## Verified hardware and software state

The following was inspected directly on the board through `ssh imx`:

| Item | Verified state |
|---|---|
| Board | NXP i.MX93 11x11 FRDM |
| Application CPU | 2x Cortex-A55, AArch64 |
| CPU INT8 feature | Arm dot-product present (`asimddp`) |
| NPU | Arm Ethos-U65, nominally 0.5 TOPS |
| NPU device | `/dev/ethosu0` |
| Device permissions | `crw------- root root`; current tests run as root |
| Reserved NPU memory | 128 MiB at `0xa8000000-0xafffffff` |
| Kernel driver | `ethosu`, device and sysfs class present |
| Firmware | `/usr/lib/firmware/ethosu_firmware` |
| Ethos-U driver/firmware stack | NXP 24.05 |
| TensorFlow Lite runtime | 2.16.2, Python module available |
| Ethos-U delegate | `/usr/lib/libethosu_delegate.so` |
| Ethos-U support library | `/usr/lib/libethosu.so.1.0.0` |
| Vela compiler | 3.12.0 at `/usr/bin/vela` |
| Vela accelerator default | `ethos-u65-256` |
| Vela configuration file | `Arm/vela.ini` |
| ONNX Runtime | 1.17.1, `CPUExecutionProvider` only |

The kernel log confirms that the 128 MiB reserved region was initialized and assigned to
the Ethos-U driver. The presence of the device, firmware and delegate means the board-side
execution stack is installed. It does **not** prove that the current model can be delegated;
there is not yet a compatible compiled model or a DroneVision TFLite runtime.

## NPU sanity check: bundled MobileNet V1

Status: complete on 7 August 2026. This validates the installed Vela/compiler, delegate,
device, firmware and RPMsg execution path independently of the DroneVision model.

The board bundles this full-integer MobileNet artifact:

| Property | Value |
|---|---|
| Original model | `/usr/bin/tensorflow-lite-2.16.2/examples/mobilenet_v1_1.0_224_quant.tflite` |
| Original SHA-256 | `ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd` |
| Input | NHWC `[1,224,224,3]`, `uint8`, scale `0.0078125`, zero point `128` |
| Output | `[1,1001]`, `uint8`, scale `0.00390625`, zero point `0` |
| Test image | bundled `grace_hopper.bmp` |

The unmodified model was compiled on the board:

```bash
vela /usr/bin/tensorflow-lite-2.16.2/examples/mobilenet_v1_1.0_224_quant.tflite \
  --accelerator-config ethos-u65-256 \
  --optimise Performance \
  --show-cpu-operations \
  --show-subgraph-io-summary \
  --verbose-performance \
  --output-dir /root/dronevision-ai/bench/out/npu-sanity
```

Vela assigned all 60 operators to one NPU subgraph and reported zero CPU operators. It
estimated 4.49 ms/inference, 222.62 inferences/s, 572,406,226 MACs, 370.91 KiB maximum
SRAM use and 3,719.84 KiB DRAM use. The compiled model is 3.3 MiB and has SHA-256
`f57cf65901827a9fb1c5917cdf859c158044a3e23385a790b64787470e6ab6c3`.

The original model was then run through TFLite/XNNPACK on one A55 thread, and the compiled
model was loaded through `/usr/lib/libethosu_delegate.so`. Each case used 10 warm-ups and
100 timed invocations of the same input:

| Runtime | Mean | Median | P95 | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| TFLite CPU, original model | 51.7906 ms | 51.6983 ms | 52.1594 ms | 51.2735 ms | 55.3803 ms |
| Ethos-U65, Vela model | 3.7754 ms | 3.7695 ms | 3.8210 ms | 3.7527 ms | 3.8782 ms |

The measured mean and median NPU speed-ups were both about **13.7x**. Top-1 was identical
(`military uniform`, class 653). Of 1,001 quantized output elements, 996 were bit-exact;
the remaining five differed by one `uint8` unit, with mean absolute delta 0.004995 and no
change in top-1. This is the expected scale of integer-backend rounding variation.

Execution evidence was explicit:

```text
Ethosu delegate: device_name set to /dev/ethosu0
EthosuDelegate: 1 nodes delegated out of 1 nodes with 1 partitions
remoteproc0: Booting fw image ethosu_firmware
rpmsg-ethosu-channel created
```

The board finished at 58.35 degrees C with 1.6 GiB memory available. There was no timeout,
delegate fallback, driver error or memory pressure. Vela's generated model and CSV reports
were copied to the local ignored directory `bench/out/npu-sanity/`.

This proves the board's NPU stack works end-to-end. It does not establish that DroneVision's
YOLO graph is compatible or fast; that still depends on raw-head export, INT8 calibration,
operator coverage and end-to-end accuracy.

## Reference artifacts

The model used for all existing results is 320x320 and has one object class.

| Artifact | SHA-256 | State |
|---|---|---|
| `models/drone_yolo26n_v4.pt` | `841edc03a993fabd57916f0622d2a66af384f66a1f73b6fd7444ed92a893ed6d` | PyTorch source weights available |
| `models/drone_yolo26n_v4.onnx` | `7acd721e718a6fe444b22eb93300d2316ae4413209c2a89789072d4e1a4a3bc9` | FP32 CPU reference |
| `models/drone_yolo26n_v4_raw.onnx` | `f43e8f0f7445a89d8dcfa1d03fa3be98ace40e165d079e2d7cd2a4aa95d1b6f7` | FP32 raw-head quantization reference |

The current ONNX graph has FP32 input `images[1,3,320,320]` and end-to-end output
`output0[1,300,6]`, where each row is `x1,y1,x2,y2,confidence,class`. That export includes
the final detection-selection head.

## Two separate INT8 tracks

INT8 CPU execution and Ethos-U execution should be treated as separate experiments. They
need different model formats and answer different questions.

### Track A: INT8 ONNX on the Cortex-A55 CPU

Create a statically quantized ONNX QDQ or QOperator model and run it with ONNX Runtime's
`CPUExecutionProvider`. This measures the benefit from the A55 `asimddp` instructions and
from reduced weight/memory traffic. It is a useful baseline and fallback, but it does not
use `/dev/ethosu0`.

Requirements:

- Use static calibration, not weights-only/dynamic quantization, for convolution-heavy
  vision inference.
- Quantize convolution and other supported compute nodes while retaining unsupported
  nodes in FP32 if necessary.
- Record the quantizer/tool versions, quantization format, tensor types, calibration
  manifest, model hash and any excluded nodes.
- Run the same `bench.speed`, `bench.parallel` and `bench.accuracy` tests as FP32.
- Confirm from the runtime description that the provider remains CPU and the artifact is
  reported as INT8.

This result is also the closest architectural comparison to future Pi 5 INT8 CPU tests,
because both the Cortex-A55 and Cortex-A76 expose Arm dot-product instructions.

### Track B: full-integer TFLite compiled for Ethos-U65

The NPU path needs a fully quantized TensorFlow Lite model, followed by an offline Vela
compile. Vela replaces supported TFLite subgraphs with an Ethos-U custom operator. At run
time, TFLite loads the compiled model and `/usr/lib/libethosu_delegate.so` sends those
subgraphs to `/dev/ethosu0`. Remaining unsupported operations execute on the CPU.

The useful performance number is therefore not merely "delegate loaded". We must record
how much of the graph Vela delegates and which operations fall back to the CPU. A small
unsupported island or repeated CPU/NPU tensor conversions can dominate end-to-end time.

## Required model re-export

Status: phase 2 complete on 7 August 2026. `python -m bench.export_raw` used the
checkpoint's original Ultralytics 8.4.104 exporter with batch 1, static 320x320 input,
opset 12, simplification enabled, `nms=False` and `end2end=False`. The resulting graph has
input `[1,3,320,320]` and decoded raw output `[1,5,2100]`. It contains no `TopK`,
`GatherElements` or `NonMaxSuppression` nodes, passes the ONNX checker and records the
source checkpoint hash in its metadata.

The existing end-to-end ONNX output is not the preferred quantization source. Static
quantization and Vela commonly reject or leave detection-head operations such as `TopK`
and `GatherElements` on the CPU. The completed export omits the end-to-end selection/NMS
head and exposes the raw decoded YOLO tensor instead:

```text
(1, 4 + number_of_classes, anchors)
```

or its transpose. For the one-class 320x320 detector this is a five-channel raw head with
2,100 anchors for strides 8, 16 and 32. DroneVision's `yolo_codec.py` already detects and
decodes both raw layouts as well as the current `(1,N,6)` end-to-end layout. It also does
confidence filtering and optional NMS on the CPU, where that small amount of work is
preferable to preventing NPU delegation.

The raw export must keep:

- static batch size 1;
- static 320x320 spatial dimensions;
- the same RGB input convention;
- pixel-space box coordinates, not normalized coordinates;
- the same one-class weights and confidence semantics;
- no embedded NMS, TopK or final detection selection.

The comparison against the end-to-end FP32 model is complete on all 337 frame sets. The
raw model uses YOLO26's trained one-to-many branch rather than the end-to-end model's
one-to-one branch, so its output is not expected to be bit-identical. It produced a strict
accuracy improvement:

| FP32 artifact | Localized | Misses | Mean 3D | Median 3D | P95 3D | Maximum 3D |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end reference | 333/337 | 4 | 36.42 mm | 32.07 mm | 77.58 mm | 181.64 mm |
| Raw-head reference | **337/337** | **0** | **30.87 mm** | **28.10 mm** | **59.13 mm** | **162.78 mm** |

The raw model averaged 980.44 ms per four-camera fix with two ONNX threads, essentially
the same CPU cost as the original graph. Isolated inference was 237.38 ms and raw decode
added approximately 0.83 ms. All subsequent INT8 accuracy comparisons must use this raw
FP32 result as their primary reference; retain the end-to-end row only for continuity with
the Pi measurements.

## Representative calibration data

Status: phase 1 complete on 7 August 2026. `data/corpus/quantization_split.json`
(SHA-256 `9745f562f99087a2a558e141b15a1f1aa587d848f9d0cdaea3750a60fb121e24`)
selects 64 evenly spaced frame sets across the complete recording, including every camera:
256 calibration images. The other 273 synchronized frame sets (1,092 images) form the
quantization holdout. The generator is `python -m bench.quant_split`; it validates source
hashes, camera completeness, ZIP membership, overlap and full-corpus coverage. All 256
selected JPEGs decoded successfully as 360x640 RGB-capable `uint8` images, and the complete
ZIP passed its CRC check.

Use real DroneVision inputs, not random tensors or generic images. Calibration samples
should be drawn deterministically from `data/corpus/frames.zip` and include all four
cameras across the full recording. Preserve the production preprocessing:

1. Decode JPEG to RGB.
2. Letterbox without changing aspect ratio to 320x320.
3. Use padding value 114.
4. Preserve the network's expected value range and dtype during conversion.

The current FP32 codec converts HWC RGB `uint8` to NCHW FP32 in `[0,1]`. A TFLite export
will normally use NHWC and may expose `int8` or `uint8` input. The new runtime must inspect
the model's input quantization scale and zero point instead of assuming that dividing by
255 is correct. Output tensors must likewise be dequantized using their declared scale
and zero point before the existing box decoder consumes them.

Keep a calibration manifest containing the exact frame-set IDs and camera names. Start
with several hundred images distributed across cameras and time; enlarge it if activation
ranges or accuracy are unstable. The validation corpus may be used for this engineering
comparison, but a separate held-out sequence will eventually be needed to make a general
accuracy claim.

## Conversion and compilation workflow

The exact exporter commands depend on the installed Ultralytics/PyTorch version and must
be captured when run. The intended artifact chain is:

```text
PyTorch `.pt`
  -> raw-head FP32 export
  -> FP32 TFLite functional reference
  -> representative-dataset full INT8 TFLite (Ultralytics FP32 public boundaries)
  -> Vela-compiled `_vela.tflite`
  -> remove only the compiled graph's boundary QUANTIZE/DEQUANTIZE nodes
  -> TFLite + Ethos-U delegate on i.MX93
```

The TFLite converter must request full integer quantization. Do not accept a model that
only quantizes weights while leaving activations FP32. Prefer integer model input/output
when supported, because it avoids CPU float/INT8 conversion around every inference.

Compile the resulting full-integer model on the board with a command of this form:

```bash
mkdir -p bench/out/vela

vela models/drone_yolo26n_raw_int8.tflite \
  --accelerator-config ethos-u65-256 \
  --optimise Performance \
  --show-cpu-operations \
  --show-subgraph-io-summary \
  --verbose-performance \
  --output-dir bench/out/vela
```

Vela 3.12.0 defaults to `ethos-u65-256`, Performance optimization and a 393,216-byte
arena cache, but these values should still be explicit in a reproducible command. The
installed generic `Arm/vela.ini` contains several example memory systems; do not select a
named `--system-config` or `--memory-mode` until it is confirmed against the i.MX93/NXP
memory integration. The first correctness test can use Vela's internal defaults.

Save the complete compiler output. In particular, preserve:

- input model hash and compiled model hash;
- Vela version and full command line;
- operator allocation and CPU fallback list;
- estimated operations, cycles, bandwidth and peak memory;
- subgraph input/output types, shapes, scales and zero points;
- warnings about unsupported operators or tensor constraints.

### Verified integer input/output boundary export

Ultralytics 8.4.104 deliberately applies `NO_QUANTIZE` to the public input and output.
Compiling that calibrated graph with Vela 3.12.0 works, but leaves the runtime graph as
`QUANTIZE (CPU) -> ethos-u -> DEQUANTIZE (CPU)`. Requantizing the FP32 model directly with
integer boundaries was rejected: it changed constant encoding and Vela failed with an
`AssertionError`. Removing the two boundary operators *before* Vela also triggers that
Vela 3.12 limitation for this NCHW graph.

The verified workflow therefore compiles the original calibrated graph first, then exposes
the already-calibrated INT8 tensors around the compiled `ethos-u` node:

```bash
.venv-export/bin/python -m bench.export_tflite --mode int8-io

# For the deployable compiled model, after the normal Vela compilation:
.venv-export/bin/python -m bench.export_tflite --mode int8-io \
  --source-int8-io bench/out/vela/drone_yolo26n_v4_raw_int8_vela.tflite \
  --output-int8-io bench/out/vela/drone_yolo26n_v4_raw_int8_io_vela.tflite
```

Verified final contract and results on the i.MX93:

| Property | Result |
|---|---|
| SHA-256 | `8c8e91d09a6ea20ed64ed128034183a5c058b35952fbff214d8a62309e711e61` |
| Input | INT8 NCHW `[1,3,320,320]`, scale `1/255`, zero point `-128` |
| Output | INT8 `[1,5,2100]`, scale `0.0078429878`, zero point `-128` |
| Delegation | `1 nodes delegated out of 1`; no CPU boundary nodes |
| Raw-output parity | Exact against the prior compiled model (`max_abs=0`) |
| Invoke latency, 100 calls | 36.84 ms mean versus 37.70 ms; 2.3% lower |
| Full corpus, confidence 0.25 | 336/337, 52.39 mm mean, 95.94 mm P95; unchanged |

## Minimal board-side delegate smoke test

The following is the intended loading pattern once a `_vela.tflite` artifact exists:

```python
from tflite_runtime.interpreter import Interpreter, load_delegate

delegate = load_delegate("/usr/lib/libethosu_delegate.so")
interpreter = Interpreter(
    model_path="bench/out/vela/drone_yolo26n_raw_int8_vela.tflite",
    experimental_delegates=[delegate],
)
interpreter.allocate_tensors()
print(interpreter.get_input_details())
print(interpreter.get_output_details())
```

Before timing, run a real corpus image, verify tensor quantization metadata and compare its
decoded boxes with the FP32 raw-head reference. Loading without an error is necessary but
is not proof of NPU execution; use the Vela allocation report and board logs/profiling to
confirm delegation.

## DroneVision integration needed

There is currently no TFLite/Ethos-U runtime implementation in DroneVision. Add one behind
the existing `TensorRuntime` contract, without changing the detector or geometry layers.
It should:

- load a normal `.tflite` model on CPU or a Vela model with an explicit delegate path;
- query input/output shape, dtype, scale and zero point from the interpreter;
- handle TFLite's expected NHWC layout while preserving RGB and letterboxing semantics;
- quantize input and dequantize raw output correctly;
- reuse preallocated input buffers where the Python TFLite API permits it;
- report model hash, runtime/delegate versions, tensor metadata and whether delegation was
  requested in `describe()`;
- fail clearly if the delegate, device or required quantization metadata is missing;
- support clean teardown and repeated benchmark construction;
- keep raw-head decoding in the shared `yolo_codec.py` implementation.

Do not silently fall back to CPU when an NPU benchmark was requested. A separate explicit
TFLite-CPU mode is useful for attribution, but an Ethos-U mode should fail if the delegate
cannot load.

## Validation sequence and gates

Change one variable at a time and retain every intermediate artifact:

1. **Raw-head FP32 parity (complete):** compare the new raw export with the current FP32
   end-to-end model on the full corpus.
2. **Uncompiled INT8 parity:** run the full-integer TFLite model on TFLite CPU. This
   isolates quantization error from Vela and the NPU.
3. **Vela compatibility:** compile it and inspect CPU fallbacks before benchmarking.
4. **Single-image NPU parity:** compare raw/dequantized outputs and decoded boxes for known
   corpus frames.
5. **Full-corpus NPU accuracy:** repeat all 337 synchronized frame sets.
6. **Isolated NPU speed:** warm up, then report mean, median, P95 and maximum inference
   time separately from preprocessing and postprocessing.
7. **Four-camera scheduling:** test one shared sequential NPU stream first, then controlled
   one- versus two-worker execution. There is one NPU, so CPU-style four-worker scaling
   must not be assumed.
8. **End-to-end timing:** include JPEG decode, quantization, NPU inference, raw-head decode
   and triangulation.
9. **Sustained run:** monitor latency drift, CPU utilization, temperature, memory and
   errors for at least 10-15 minutes.

The original end-to-end reference is:

```text
337 frame sets
333 localized / 4 misses
36.42 mm mean error
32.07 mm median error
77.58 mm P95 error
181.64 mm maximum error
```

The primary quantization reference is now the raw FP32 result: 337 localized, zero misses,
30.87 mm mean, 28.10 mm median, 59.13 mm P95 and 162.78 mm maximum error.

Accuracy and performance must be reported together. The original engineering notes used
a provisional 10% degradation gate, but it is not an established project requirement and
should not be presented as one. Report isolated images/s and complete four-camera fix
latency so scheduling effects remain visible.

Applied to the new raw FP32 reference, the provisional INT8 gate is:

```text
localized        337/337 raw FP32 reference
mean 3D error    30.87 mm raw FP32 reference
P95 3D error     59.13 mm raw FP32 reference
complete fix     report measured latency and fixes/s
camera inference report measured latency and images/s
```

## Next steps: phase 3 full-integer TFLite

The NPU track is the priority. Static INT8 ONNX for the A55 remains useful for attribution
and Pi 5 comparison, but it should not delay the TFLite/Vela path.

### 3.1 Prepare the pinned conversion environment

1. Recreate an isolated Python 3.12 exporter environment.
2. Pin Ultralytics `8.4.104`, matching the checkpoint and raw ONNX export.
3. Capture the exact Torch, TensorFlow/LiteRT, ONNX and conversion-tool versions.
4. Check the installed exporter's LiteRT requirements before downloading the larger
   TensorFlow conversion stack; record any host-platform restriction.

Do not keep the generated environment in Git. The previous temporary export environment
was removed after phase 2; only the command/version record and generated artifacts belong
in the repository.

### 3.2 Materialize the representative images without changing bytes

Add a small utility that reads `data/corpus/quantization_split.json` and extracts exactly
its 256 calibration JPEG members into ignored `data/scratch/` storage. It must:

- verify the three corpus hashes from the manifest before extraction;
- preserve original JPEG bytes rather than decoding and re-encoding;
- preserve camera and frame-set identity in paths or a generated index;
- verify 64 frame sets, four cameras each, 256 unique files;
- produce a minimal dataset YAML or image list accepted by the pinned exporter.

Labels are not required to estimate activation ranges, but the converter must demonstrably
consume the selected images rather than silently falling back to a default sample dataset.

### 3.3 Export an FP32 TFLite functional reference

Before introducing quantization, export `end2end=False`, batch-1, static 320x320 FP32
TFLite/LiteRT. Inspect its input layout and output shape, then run several corpus images on
TFLite CPU. Decoded boxes should agree closely with `drone_yolo26n_v4_raw.onnx`. This
isolates ONNX-to-TFLite/export differences from INT8 calibration loss.

Stop here if the FP32 TFLite output layout, coordinates or detections do not match the raw
ONNX semantics.

### 3.4 Export full-integer TFLite

Run representative-dataset static quantization using only the 256 manifested images.
Require and record:

- integer weights and activations, not weights-only quantization;
- preferably `int8` or `uint8` model input and output;
- static `[1,320,320,3]` NHWC input unless the exporter proves otherwise;
- raw decoded `[1,5,2100]` output or a documented transpose;
- every input/output scale and zero point;
- exporter command, dependency versions, calibration-manifest hash and model SHA-256;
- an operator/dtype inventory proving that unexpected FP32 islands are absent.

### 3.5 Quantization-only correctness gate

Run the uncompiled INT8 model with TFLite CPU before Vela. Compare the same images in this
order:

1. individual calibration and holdout frames at tensor and decoded-box level;
2. the 273-set holdout for the primary quantization gate;
3. all 337 frame sets for continuity with the existing tables.

If the TFLite CPU result fails the accuracy gate, correct calibration/export first. Vela
cannot recover accuracy already lost in the quantized artifact.

### Phase-3 exit criteria

- [x] FP32 TFLite artifact has raw-ONNX functional parity (exact 30.87 mm mean 3D error, 0 misses).
- [x] Full-integer TFLite consumes the committed 256-image calibration selection.
- [x] Tensor dtypes, shapes, scales and zero points are recorded.
- [x] INT8 TFLite CPU evaluated on calibration (56.18 mm), holdout (53.59 mm), and full corpus (54.08 mm, 0 misses).
- [x] Commands, tool versions and artifact hashes are reproducible (`bench.export_tflite`, `bench.eval_quant_gate`).
- [x] The INT8 artifact is ready for Vela operator-coverage analysis.

## Benchmark matrix

At minimum retain these comparable rows:

| Artifact/runtime | Device | Purpose |
|---|---|---|
| Existing FP32 ONNX | A55 CPU | Immutable reference |
| Raw-head FP32 ONNX or TFLite | CPU | Export/decoder parity |
| Static INT8 ONNX | A55 CPU | `asimddp` and memory-traffic benefit |
| Full INT8 TFLite, uncompiled | A55 CPU | Quantization-only reference |
| Vela INT8 TFLite with delegate | Ethos-U65 plus any fallback | Final NPU result |

All rows must use batch 1, 320x320, the same RGB/letterbox preprocessing, the same corpus,
the same confidence and IoU thresholds and the same accuracy calculation. Record model
hashes rather than relying on filenames.

## Known risks and likely failure modes

- **End-to-end detection head is not delegatable.** Use the raw output head and CPU decode.
- **Partial quantization.** A small number of FP32 activations can prevent large regions
  from reaching the NPU; inspect the generated model and Vela report.
- **Unsupported TFLite operators or tensor constraints.** Replace/export them differently
  rather than accepting a large CPU island unnoticed.
- **Incorrect input layout or color order.** Current source frames are RGB; TFLite is
  likely NHWC while ONNX is NCHW.
- **Incorrect scale/zero point.** Always read tensor metadata and test saturation/range.
- **Quantized output decoded as FP32.** Dequantize before raw-head box decoding.
- **Calibration under-represents cameras or target sizes.** Use all cameras and the whole
  recorded time range with a saved manifest.
- **Delegate loaded but little work delegated.** Treat Vela operator coverage as a primary
  result.
- **Too many NPU workers.** A single accelerator may serialize requests while duplicating
  buffers and increasing contention; benchmark rather than assuming CPU topology rules.
- **Model size mistaken for speed.** INT8 usually reduces storage and bandwidth, but the
  real metric is complete fix latency.
- **Permissions.** `/dev/ethosu0` is currently root-only. Add a deliberate udev/group rule
  before running DroneVision as an unprivileged service; do not broadly chmod the device.
- **Version coupling.** Vela output, delegate, firmware and driver need compatible command
  stream versions. Keep the board's NXP stack together and capture versions in results.
- **Nominal TOPS mistaken for delivered throughput.** The 0.5-TOPS rating does not predict
  YOLO latency or include CPU fallback, transfer, preprocessing and postprocessing.

## Immediate preparation checklist

- [x] Pin/export the raw-head FP32 model from the existing `.pt` weights.
- [x] Prove raw-head FP32 accuracy parity or improvement on all 337 frame sets.
- [x] Create a representative calibration manifest (ready to commit/version).
- [x] Produce full-integer TFLite and static INT8 ONNX artifacts separately.
- [x] Record converter versions, commands, hashes and tensor metadata.
- [x] Compile TFLite with Vela and archive the complete operator/fallback report.
- [x] Implement and test a TFLite/Ethos-U `TensorRuntime` backend.
- [ ] Confirm actual delegation before collecting performance numbers.
- [x] Run accuracy before optimizing scheduling.
- [ ] Benchmark isolated inference, complete fixes and sustained operation.

A bundled INT8 MobileNet has now been compiled and executed successfully on the Ethos-U65,
proving the board stack. No DroneVision INT8 model has yet been generated or compiled, and
no DroneVision inference has yet run on the NPU. The existing FP32 CPU measurements remain
the baseline against which those future results must be attributed.
