# dronevision — Arm-optimized multi-camera 3D localization

Locate a flying drone in 3D from fixed cameras — **no GPS, no onboard sensing** — fast
enough to close a control loop on a Raspberry Pi.

**License: Apache-2.0** · Arm Create: AI Optimization Challenge 2026, Physical AI track

> 🚧 **In progress.** The pipeline runs on a Raspberry Pi 4 and the first Arm numbers are
> in (below). Quantization, the Pi 5 comparison and KleidiAI are still outstanding.

## The question this project answers

The same CNN, on two Raspberry Pis one generation apart, is not the same optimization
problem. A Pi 4 (Cortex-A72, Armv8.0-A) has no dot-product instructions; a Pi 5
(Cortex-A76, Armv8.2-A) has `FEAT_DotProd`. Int8 quantization — the reflexive first move in
edge AI — pays very differently across that boundary, and most guidance does not say so.

So this repository does two things:

1. **Ships a real workload**, not a microbenchmark: four cameras, a live 3D position, a
   drone that actually flies on the output.
2. **Attributes the speedup.** Not "we got N× faster" but *which layer each multiplier came
   from*, which are architecture-dependent, and which are free.

## Measured results

Current result on the shipped corpus with the **colour-marker** detector
(`py -m bench.accuracy --site factory --detector color`). This is a reference for the
optimization work, not an AI target:

| | value |
|---|---|
| 3D error vs ground truth | **6.28 mm** mean · 6.19 median · 10.87 p95 · 14.79 maximum |
| with 2 of 4 cameras occluded | 22.19 mm mean — degrades, keeps working |
| pipeline cost | ~2.2–2.8 ms → 360–450 Hz compute ceiling |
| with JPEG decode included | ~4.0 ms → ~250 Hz |

The current nominal factory geometry reproduces all 337/337 localizations. It keeps the
detector-to-vehicle offset explicit instead of absorbing it into the camera extrinsics.

Per-stage, in milliseconds (one representative run):

```
acquire      0.007      (1.683 when frames arrive compressed)
detect       1.924      <- the layer the Arm work targets
associate    0.003
triangulate  0.300
smooth       0.005
```

`detect` is the whole story: swap the colour marker for YOLO and it goes from ~2 ms to tens
of milliseconds, and that single row becomes the benchmark below.

**Accuracy figures are exact and reproducible** — the corpus is a fixed set of bytes, so any
machine gets the same millimetres. **Timing figures are not.** The spread above is real: the
same command measured 2.23 ms on an idle host and 2.79 ms with the simulator, 3D viewer and
AI process running, repeatable to ±0.03 ms within either state. Two consequences, and both
matter because timing *is* the deliverable:

- Never compare a timing across machine states — only within one run of the sweep;
- The Arm numbers will be measured on an otherwise-idle board with thread pinning, and the
  conditions reported alongside them.

### YOLO26n on Arm — measured

One 320×320 forward pass of the same artifact (`drone_yolo26n_v4.onnx`, sha `7acd721e718a`),
ONNX Runtime 1.28, `CPUExecutionProvider`, spin-wait disabled. Regenerate with
`python -m bench.report`.

| Board | Core | dotprod | Precision | Threads | Inference ms | inf/s |
|---|---|---|---|---|---|---|
| Pi 4 Model B | Cortex-A72 | **no** | fp32 | 1 | 205.4 | 4.8 |
| Pi 4 Model B | Cortex-A72 | **no** | fp32 | 4 | **93.7** | 10.4 |
| x86-64 desktop | — | n/a | fp32 | 1 | 12.0 | 78.2 |
| x86-64 desktop | — | n/a | fp32 | 4 | 5.1 | 171.2 |

- **Pi 4 is 18.4× slower** than the desktop at the same thread count.
- **Thread scaling is 2.19×, not 4×** — the kernel is memory-bound, so cores are not the
  lever. Four cameras at 93.7 ms each is 375 ms per fix, or 2.7 Hz: too slow to close a
  loop, which is the problem the rest of the work exists to solve.
- **Accuracy is architecture-independent**: 36.42 mm 3D error on the Pi, identical to the
  desktop to the last digit. Same weights, same codec, same corpus.

### Optimization 1 — multi-core parallelism

A fix needs all four cameras. Four cores can be spent two ways, and they are not
equivalent: split one image across all cores, or give each core a whole image and a
single-threaded engine. Measured on the Pi 4, three interleaved passes of 20 fixes each,
best pass taken (`python -m bench.parallel --iters 20 --repeat 3`):

| Strategy | ms per fix | Hz | per-worker inference | vs sequential |
|---|---|---|---|---|
| 1 worker × 4 threads | 376.6 | 2.66 | 92 ms | 1.00× |
| 2 workers × 2 threads | 345.8 | 2.89 | 168 ms | 1.09× |
| **4 workers × 1 thread** | **322.5** | **3.10** | 311 ms | **1.17×** |

**1.17×, not the ~3× a core count suggests.** The reason is in the per-worker column, and
it is the most useful thing measured so far. Scaling the worker count with one thread each:

| workers | wall ms/fix | speedup | efficiency | per-worker inference |
|---|---|---|---|---|
| 1 | 828.6 | 1.00× | 100% | 205 ms |
| 2 | 466.1 | 1.78× | 89% | 229 ms |
| 3 | 476.6 | 1.74× | 58% | 252 ms |
| 4 | 321.5 | 2.58× | 64% | 311 ms |

Each added worker makes *every* worker slower — 205 → 311 ms — while wall time still
falls. So the work genuinely overlaps and something shared is saturating. It is not the
GIL: under lock contention each worker's own inference would stay at 205 ms and the wall
time would not improve at all. It is memory bandwidth, on a single-channel LPDDR4 SoC.

Three consequences:

- **Multiprocessing is measurably worse**, not merely unnecessary. Both backends are
  implemented and measured (`--backends thread,process`):

  | backend | 4 workers × 1 thread | vs threads |
  |---|---|---|
  | thread | **319.9 ms** | — |
  | process | 358.7 ms | **12% slower** |

  The usual reason to reach for processes is the GIL, and the GIL is not the constraint:
  ONNX Runtime releases it during inference, so threads already overlap. What processes
  add is a 691 KB frame pickled down a pipe per camera — 2.8 MB of extra copying per fix
  on a board whose bottleneck is *already* memory traffic. Each process worker's own
  inference is slower too (330 ms against 309 ms), which is the same bus contention
  showing up again. Adding memory traffic to a memory-bound workload makes it worse.
- **The 3-worker row is worse than the 2-worker row** because four cameras do not divide
  by three: one worker does two inferences and the fix waits for it. Worker count should
  divide the camera count.
- **This reorders the remaining optimizations.** If the bottleneck is bytes moved rather
  than cores available, then int8 (4× less weight traffic), lower input resolution and
  ROI cropping are the primary levers — not more parallelism. Quantization stops being
  "modest on a core without dotprod" and becomes the main event.

Accuracy is unchanged by any of this: **36.42 mm either way**, to the digit.

```bash
python -m dronevision.service --detector yolo --runtime onnx --parallel
```

Still pending: int8 (a raw-head re-export is needed first — see below), the Pi 5 column,
and ExecuTorch + KleidiAI.

| | Pi 4 (A72) | Pi 5 (A76) |
|---|---|---|
| ONNX Runtime int8 | pending | pending |
| ExecuTorch + KleidiAI int8 | n/a — no dotprod | pending |
| *+ camera scheduling* | pending | pending |

**The Pi 4 has no `asimddp`** — verified on the board, not assumed:
`Features : fp asimd evtstrm crc32 cpuid`. So int8 there gets NEON only and KleidiAI's
SDOT kernels never engage. That is the measurement the Pi 5 column exists to contrast with.

## Architecture

The package *is* the diagram. Directory names carry their ordinal, so a listing reads
top-to-bottom in processing order instead of alphabetically:

```
dronevision/
├── l1_image_sync/      1. Image Synchronization
├── l2_perception/      2. Perception Layer (YOLO)
├── l3_association/     3. 2D Detection Association
├── l4_triangulation/   4. DLT Triangulation
├── l5_estimation/      5. EKF State Estimation and Filtering
├── io/                 boundaries: frames in, state estimate out
├── pipeline.py         chains the five layers
└── service.py          runs the block as a process
```

| Layer | What it does | State |
|---|---|---|
| 1 `l1_image_sync` | frame sets + **which camera the detector looks at** | 39 ms mean camera skew measured; scheduler pending |
| 2 `l2_perception` | per-camera detection — the expensive layer | 4 detectors; Arm runtimes are the optimization target |
| 3 `l3_association` | cross-camera association + per-camera Kalman coasting | coasting works; multi-target pending |
| 4 `l4_triangulation` | camera geometry + multi-view DLT | done, with consensus outlier rejection |
| 5 `l5_estimation` | position filtering | EMA; EKF3D scaffolded — see below |

`tests/test_boundary.py` asserts this layout matches `dronevision.LAYERS`, that no sixth
layer appears, and that no layer imports a later one.

## Scope: this repository is the AI block

The drone, the cameras and the control software belong to the companion simulator project
and are **consumed as network services**, never copied:

```
  simulator project                        this project
  ┌──────────────────┐   ZeroMQ, JPEG    ┌──────────────────┐
  │ cameras          ├──── :5555 ───────►│ AI: 5 layers     │
  │ drone (PX4)      │                   │                  │
  │ control software │◄─── :5601 ────────┤ state estimate   │
  └──────────────────┘   UDP, JSON       └──────────────────┘
```

The two share **no code and no filesystem**. This repo references the simulator in one
optional test; the simulator references this repo in one line of help text.

### Design rule

**Nothing under `dronevision/` imports Gazebo, ROS, PX4 or MAVLink.** That is why this
installs on a bare Raspberry Pi, and why the benchmarks run with no simulator anywhere in
sight. There is no simulator bridge and no control software here — both belong to the other
project, and a test enforces it by parsing every module.

## Install

```bash
git clone https://github.com/SCAI-Engineering/dronevision-ai.git && cd dronevision-ai
pip install -e ".[pi]"        # numpy, opencv, pyyaml, pyzmq, onnxruntime
pytest                        # 191 tests, no hardware needed
```

Extras: `net` (transport) · `ort` (ONNX Runtime) · `torch` (reference backend + export) ·
`pi` (= net + ort) · `dev` (tests).

## Running it

**Offline** — no simulator, no drone, no network. Also exactly what runs on a Pi:

```bash
py -m bench.accuracy                       # accuracy + per-stage timing
py -m bench.accuracy --detector yolo --runtime onnx --threads 4
py -m bench.accuracy --occlude cam_ne,cam_sw
py -m bench.accuracy --mode encoded        # include JPEG decode in the timing
py -m bench.speed --runtime onnx --threads 4 --json bench/out/pc.json
py -m bench.report                         # render the results table
py -m dronevision.service --replay data/corpus
```

**On a Raspberry Pi**, driven from this machine — the board needs no GitHub access:

```bash
./bench/pi.sh info      # board, cores, CPU features, temperature
./bench/pi.sh setup     # ship the tree, build the venv, install deps
./bench/pi.sh bench     # sweep, then pull the JSON back here
./bench/pi.sh run -m bench.accuracy --detector yolo --runtime onnx --threads 4
```

**Live**, against the simulator:

```bash
# simulator host
./services.sh up && ./drone.sh takeoff

# here — the same command on a laptop or a Raspberry Pi
py -m dronevision.service --frames tcp://<host>:5555 --state <host>:5601
py -m bench.record --seconds 30            # capture a new corpus
```

## The corpus

`data/corpus/` — 337 frame sets, 30 s, ~13 MB, four cameras, JPEG q90, with ground truth
time-aligned to each frame set. A directory of `meta.json` + `manifest.jsonl` + `frames.zip`:
one zip ships in git and stays inspectable, JSON Lines survives a truncated write, nothing is
pickled.

It exists so every claim above is reproducible on any machine with nothing else installed —
which is what makes benchmarking on an Arm board, or by anyone else, possible at all.

**Known limitation: it is hover-only.** The vehicle moves ~10 cm horizontally, so it does not
exercise tracking, coasting, or motion-dependent detectors — `--detector motion` finds
nothing on it, because background subtraction needs a moving target. A trajectory corpus is
the next thing it needs.

## What measuring turned up

Findings that shaped the code, kept here because each one is a trap:

- **Ground truth is not the obvious pose.** The simulator's pose topic publishes an entry per
  model *and* per link. The link entry looks authoritative and never moves — it is a static
  model-relative pose. Using it silently makes every number wrong.
- **The marker offset is not the one in the model file.** The file says 0.18 m above the
  parent link; measured against the pose actually reported as truth it is **0.4334 m**,
  constant across a 2.26 m climb. Taking the file value puts every altitude 253 mm out, while
  leaving horizontal accuracy untouched — so it reads as a calibration quirk, not a bug.
- **Bundle adjustment must use the point the detector sees.** For the colour detector, pass
  `--marker-offset` to `bench.refine_calibration` so the optimizer compares marker pixels with
  marker-height ground truth. Without it, the offset can be absorbed into camera extrinsics:
  reprojection improves while the physical calibration and triangulation can get worse. The
  current factory config therefore uses nominal simulator poses plus an explicit 0.4333 m
  detector offset.
- **JPEG barely costs accuracy, and saves 70× the bandwidth.** q90 measures 3.84 mm against
  4.22 mm for raw frames (mild low-pass stabilises a 7 px blob's centroid), at 6 Mbit/s
  versus 442. It collapses below q80.
- **Single-pass outlier rejection fails on a gross outlier.** Judging views against a solution
  the outlier already corrupted flags the *good* cameras. Replaced with minimal-subset
  consensus; the old path is kept as `method="single_pass"` with a test pinning its failure.
- **A stale frame is worse than no frame.** When the camera service died, the AI emitted a
  confident frozen position indefinitely, and the control software injected it. Frames older
  than `max_age` are now reported as absent, per camera, with automatic recovery.
- **Camera coverage is not total.** 85.4% of the room volume is seen by ≥2 cameras; the rest
  is single-camera dead zone at the corners. One pixel of detection error is ~2 cm in the
  interior and ~3.5 cm at the edges.

### On layer 5 (`l5_estimation`)

Layer 5 is an exponential moving average, not an EKF. The EKF is scaffolded and unwired.
That is a design choice: the flight controller downstream runs its own EKF and does the real
fusion against inertial data, so a second full estimator here would duplicate it. What this
layer owes the controller is a smooth, outlier-free position and an honest statement of how
stale it is.

## Companion project

The Gazebo/PX4 simulator supplying the world, the drone, the cameras and the flight control
is a separate project. It is the test bench and the service provider; this repository is the
deployable AI. Both wire formats are owned and documented here — the frame transport in
[dronevision/io/sources/net.py](dronevision/io/sources/net.py) and the state estimate in
[dronevision/io/schema.py](dronevision/io/schema.py) — so either service can be reimplemented
against them, including by real hardware.

## Documentation site

The project wiki is built with Material for MkDocs. Preview it locally without publishing:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000`. GitHub Pages deployment is defined in
`.github/workflows/docs.yml` and runs only after the documentation changes are pushed to
`main` and Pages is enabled for GitHub Actions in the repository settings.
