# ROI benchmark: Raspberry Pi 5 vs NXP i.MX93 (CPU and NPU)

Three-way hardware comparison for the EKF-guided dynamic ROI investigation: the
same skip/crop matrix (configs A/B/C/D), the same 12 recorded trajectories, run
on the Raspberry Pi 5's CPU, the i.MX93's CPU, and the i.MX93's Ethos-U65 NPU.

## Executive summary

Cropping the detector's input to a Kalman-predicted window saves no compute on
any of the three platforms — config C is slower than the full-frame baseline
on all 36 platform/trajectory combinations tested. Frame-skipping is the real
lever everywhere: a genuine 1.7–5.0× throughput win, present on all three
platforms, that tracks the same per-trajectory pattern (clean translating
motion benefits most, pure rotation least) regardless of hardware. The i.MX93's
CPU alone is roughly 8× slower than the Pi 5's and cannot reach the 12 Hz
target under any configuration; its NPU changes that story, but only once
frame-skipping is layered on top — the NPU alone (6 Hz) falls short, NPU **plus**
skip clears 12 Hz on 10 of 12 trajectories, peaking at 25 Hz. The Pi 5's CPU
still outperforms the i.MX93's NPU throughout, by a margin that does not close
when skip is applied to both.

## Provenance

| Item | Value |
|---|---|
| Code state | `dev`, `9d3c472` and the calibration fix merged ahead of it (`92c24f8`) |
| Site calibration | `marker_dz = 0.4333`, un-tilted extrinsics — confirmed correct on all three platforms |
| Trajectories | 12 recorded motion profiles, `tools/dev/teleport_path.py` corpora, ~190–240 frame sets each |
| CPU model | `models/drone_yolo26n_v4.onnx`, FP32, SHA-256 `7acd721e718a6fe444b22eb93300d2316ae4413209c2a89789072d4e1a4a3bc9` (same artifact cited in `IMX93_VALIDATION.md`) |
| NPU model | `bench/out/vela/drone_yolo26n_v4_raw_int8_io_vela.tflite`, INT8 raw-head, Vela-compiled, SHA-256 `8c8e91d09a6ea20ed64ed128034183a5c058b35952fbff214d8a62309e711e61` |
| Pi 5 | 4× Cortex-A76, `--threads 4` |
| i.MX93 CPU | 2× Cortex-A55, `--threads 2` |
| i.MX93 NPU | Ethos-U65, `--runtime tflite --parallel 2` (2 CPU workers feeding the NPU) |

## Setup

`TrackedCamera` separates two independent levers, tested in isolation and
combined:

| # | Skip frames | Crop to prediction | Isolates |
|---|---|---|---|
| A | no | no | baseline — full-frame, every camera, every frame |
| B | yes (1 in 5) | no | the skip lever alone |
| C | no | yes | the crop lever alone — same call count as A, smaller input |
| D | yes | yes | both combined |

!!! warning "The NPU path uses a different model — accuracy figures are not comparable across platforms"
    The Ethos-U65 requires INT8 weights and activations. The NPU rows in this
    document run the quantized, raw-head TFLite export, not the FP32 ONNX
    graph used on both CPUs. **Timing** is still a fair three-way comparison —
    "how fast" does not depend on which model produced the answer. **Accuracy**
    is not: only NPU-internal A/B/C/D comparisons are meaningful for error, and
    the CPU-vs-NPU accuracy row below exists to show the size of that gap, not
    to declare a winner.

## Accuracy

Accuracy is deterministic and identical on both CPU platforms (same corpus,
same model, same frame order) — verified to the millimetre across all 48
Pi 5/i.MX93-CPU combinations. Config A, mean 3D error:

| Trajectory | CPU (both boards) | NPU (INT8, different model) |
|---|---:|---:|
| square | 49.5 mm | 82.2 mm |
| square_fast | 95.3 mm | 141.7 mm |
| linear_slow | 37.5 mm | 69.4 mm |
| linear_fast | 93.2 mm | 101.1 mm |
| ramp_smooth | 54.8 mm | 82.8 mm |
| ramp_step | 41.7 mm | 69.0 mm |
| updown_smooth | 39.1 mm | 34.6 mm |
| updown_step | 29.4 mm | 41.0 mm |
| hover_jitter | 45.7 mm | 79.5 mm |
| pitch | 42.6 mm | 62.3 mm |
| spin_roll | 96.4 mm | 99.2 mm |
| spin_yaw | 103.7 mm | 102.1 mm |

The NPU path is worse on 10 of 12 trajectories and closer on the other two —
consistent with quantization noise on a smaller INT8 model rather than a
motion-dependent effect. This is a separate question from the timing results
below and does not bear on either ROI finding.

## Throughput, three platforms

`TOTAL` compute ceiling (Hz), nothing else running on any board.

### Config A — baseline

| Trajectory | Pi 5 CPU | i.MX93 CPU | i.MX93 NPU |
|---|---:|---:|---:|
| square | 9 | 1 | 6 |
| square_fast | 8 | 1 | 6 |
| linear_slow | 8 | 1 | 6 |
| linear_fast | 8 | 1 | 6 |
| ramp_smooth | 8 | 1 | 6 |
| ramp_step | 8 | 1 | 6 |
| updown_smooth | 8 | 1 | 6 |
| updown_step | 8 | 1 | 6 |
| hover_jitter | 8 | 1 | 6 |
| pitch | 8 | 1 | 6 |
| spin_roll | 8 | 1 | 6 |
| spin_yaw | 8 | 1 | 6 |

### Config B — frame-skip

| Trajectory | Pi 5 CPU | i.MX93 CPU | i.MX93 NPU |
|---|---:|---:|---:|
| square | 38 | 5 | 23 |
| square_fast | 37 | 5 | 22 |
| linear_slow | 38 | 5 | 24 |
| linear_fast | 39 | 5 | 24 |
| ramp_smooth | 40 | 5 | 24 |
| ramp_step | 39 | 5 | 24 |
| updown_smooth | 26 | 3 | 17 |
| updown_step | 27 | 3 | 17 |
| hover_jitter | 38 | 5 | 24 |
| pitch | 32 | 4 | 24 |
| spin_roll | 16 | 2 | 11 |
| spin_yaw | 16 | 2 | 10 |

### Config C — crop

| Trajectory | Pi 5 CPU | i.MX93 CPU | i.MX93 NPU |
|---|---:|---:|---:|
| square | 7 | 1 | 5 |
| square_fast | 7 | 1 | 5 |
| linear_slow | 8 | 1 | 6 |
| linear_fast | 8 | 1 | 6 |
| ramp_smooth | 8 | 1 | 6 |
| ramp_step | 8 | 1 | 6 |
| updown_smooth | 7 | 1 | 5 |
| updown_step | 7 | 1 | 5 |
| hover_jitter | 8 | 1 | 6 |
| pitch | 8 | 1 | 6 |
| spin_roll | 6 | 1 | 4 |
| spin_yaw | 6 | 1 | 5 |

### Config D — skip + crop

| Trajectory | Pi 5 CPU | i.MX93 CPU | i.MX93 NPU |
|---|---:|---:|---:|
| square | 31 | 4 | 19 |
| square_fast | 28 | 3 | 17 |
| linear_slow | 37 | 5 | 25 |
| linear_fast | 35 | 4 | 23 |
| ramp_smooth | 40 | 5 | 25 |
| ramp_step | 39 | 5 | 25 |
| updown_smooth | 19 | 2 | 13 |
| updown_step | 18 | 2 | 13 |
| hover_jitter | 36 | 4 | 25 |
| pitch | 37 | 5 | 25 |
| spin_roll | 12 | 2 | 9 |
| spin_yaw | 14 | 2 | 9 |

Cross-validation: this benchmark's `square` config A measured 971.9 ms/fix on
the i.MX93 CPU and 159.97 ms/fix on the i.MX93 NPU. A colleague's independently
built baselines for the same board measured 977.7 ms (FP32 CPU) and land in the
same 159–168 ms band for the submitted Ethos-U65 path (`optimization/results.md`).
Two methodologies, single-digit milliseconds apart, on both paths.

## Finding 1 — cropping saves no compute, on all three platforms, 36/36

The ONNX/TFLite graph's input shape is fixed at export/compile time — `[1, 3,
320, 320]` for the CPU model, statically compiled by Vela for the NPU. Every
image is resized to that shape before inference runs, so cost tracks the
tensor size, not how many pixels were cropped away first. Config C is slower
than config A on **every trajectory on every platform**:

| Trajectory | Pi 5 CPU: A→C (ms) | i.MX93 CPU: A→C (ms) | i.MX93 NPU: A→C (ms) |
|---|---|---|---|
| square | 115.5 → 141.6 | 971.9 → 1160.0 | 156.3 → 199.8 |
| square_fast | 118.2 → 143.8 | 970.9 → 1169.1 | 156.4 → 205.2 |
| linear_slow | 117.5 → 123.6 | 975.6 → 1006.1 | 156.5 → 176.7 |
| linear_fast | 118.0 → 124.2 | 975.2 → 1011.5 | 156.4 → 176.7 |
| ramp_smooth | 119.4 → 123.0 | 973.8 → 1001.5 | 156.5 → 176.6 |
| ramp_step | 118.4 → 122.3 | 974.9 → 998.0 | 156.5 → 176.7 |
| updown_smooth | 119.6 → 144.6 | 975.3 → 1177.2 | 156.4 → 205.5 |
| updown_step | 119.4 → 145.8 | 980.5 → 1180.3 | 156.4 → 207.0 |
| hover_jitter | 119.2 → 127.8 | 974.8 → 1030.6 | 156.4 → 176.7 |
| pitch | 119.3 → 122.6 | 978.6 → 995.8 | 156.4 → 176.8 |
| spin_roll | 119.4 → 156.6 | 974.3 → 1269.4 | 156.2 → 231.2 |
| spin_yaw | 120.6 → 161.0 | 974.7 → 1301.7 | 156.2 → 217.9 |

A crop miss forces a full-frame retry — a second inference on the same tick.
The miss count is identical across all three platforms (a property of the
corpus and the tracker, not the hardware), so the extra cost is purely what
each platform charges for a wasted inference. The i.MX93's CPU pays the most
per miss (fewer cores), the Pi 5 the least; the NPU sits in between, closer to
the Pi 5 in absolute terms despite running on the same board as the far more
expensive CPU path.

## Finding 2 — frame-skip works everywhere, same per-trajectory pattern

| Trajectory | Pi 5 CPU B/A | i.MX93 CPU B/A | i.MX93 NPU B/A |
|---|---:|---:|---:|
| square | 4.2× | 5.0× | 3.8× |
| square_fast | 4.6× | 5.0× | 3.7× |
| linear_slow | 4.8× | 5.0× | 4.0× |
| linear_fast | 4.9× | 5.0× | 4.0× |
| ramp_smooth | 5.0× | 5.0× | 4.0× |
| ramp_step | 4.9× | 5.0× | 4.0× |
| updown_smooth | 3.2× | 3.0× | 2.8× |
| updown_step | 3.4× | 3.0× | 2.8× |
| hover_jitter | 4.8× | 5.0× | 4.0× |
| pitch | 4.0× | 4.0× | 4.0× |
| spin_roll | 2.0× | 2.0× | 1.8× |
| spin_yaw | 2.0× | 2.0× | 1.7× |

The ranking is identical on all three platforms: clean translating motion
clusters at 3.7–5.0×, vertical motion (`updown_*`) drops to 2.8–3.4×, pure
rotation (`spin_*`) bottoms out at 1.7–2.0×. That floor is the tracking
schedule itself — the target is often outside enough cameras' view, forcing
detection regardless of hardware — not a platform limitation.

The NPU's speedup is consistently the smallest of the three, by design rather
than by defect: config A's `detect` is already cheap on the NPU (156 ms)
relative to the pipeline's fixed costs (triangulation, smoothing), so skipping
it removes proportionally less of the total than it does on either CPU path,
where `detect` dominates completely. This is the expected signature of
Amdahl's law, not evidence that skip works worse on the NPU in absolute terms
— it still delivers up to 4× on top of NPU delegation.

## Meeting the 12 Hz target

| Platform | A | B | C | D |
|---|---:|---:|---:|---:|
| Pi 5 CPU | 0/12 | **12/12** | 0/12 | **12/12** |
| i.MX93 CPU | 0/12 | 0/12 | 0/12 | 0/12 |
| i.MX93 NPU | 0/12 | **10/12** | 0/12 | **10/12** |

No platform reaches the target on full-frame detection alone. The i.MX93 CPU
never reaches it under any configuration — cropping cannot rescue it because
Finding 1 holds there too. The i.MX93 NPU reaches it only once frame-skipping
is added (10/12, capped by the two pure-rotation trajectories at the tracking
schedule's own limit, not the hardware's). The Pi 5 reaches it on every
trajectory the moment skip is enabled. Peak throughput observed: 40 Hz (Pi 5),
5 Hz (i.MX93 CPU), 25 Hz (i.MX93 NPU) — all on config D or B.

## Conclusions

- **Do not pursue crop-based ROI on any of these three platforms as currently
  exported.** 36/36 combinations show it costing time, not saving it, and the
  cost is worst on the platform that can least afford it (the i.MX93 CPU).
  Fixing this needs a dynamic-shape export, not a hardware change — the
  blocker is how the graph is frozen, not which chip runs it.
- **Frame-skipping is the one lever that delivers, on every platform tested.**
  Budget its accuracy cost (documented separately in the ROI investigation)
  against the target's expected apparent motion, not against flight-profile
  cornering — the ranking above shows rotation and vertical motion cost more
  than hard cornering does, consistently across hardware.
- **The i.MX93 needs its NPU to be viable for this workload at all**, and even
  then needs frame-skipping layered on top to clear the 12 Hz target on most
  trajectories. Its CPU alone is not a candidate under any configuration in
  this matrix.
- **The Pi 5 outperforms the i.MX93's NPU on raw throughput throughout**, with
  or without frame-skip enabled on both. Choosing the i.MX93 over the Pi 5 for
  this pipeline has to be justified by something other than speed — power
  envelope, industrial I/O, CAN-FD, EdgeLock — because it does not win the
  speed comparison even with its accelerator engaged.
