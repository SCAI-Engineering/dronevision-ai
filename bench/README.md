# `bench/` — measurement tools

Everything here answers a question with a number. Nothing here is imported by the
`dronevision` package; these are the tools that check whether it is right and how
fast it is.

## INT8 calibration split

`python -m bench.quant_split` validates the recorded ZIP and writes
`data/corpus/quantization_split.json`. The default selects 64 frame sets evenly across the
whole recording and includes all four cameras from each: 256 representative calibration
images. The remaining 273 synchronized frame sets form the quantization holdout. No JPEGs
are copied or re-encoded, and source hashes make the manifest self-identifying.

Two kinds, and the difference matters:

| | Needs a simulator? | Purpose |
|---|---|---|
| `validate_live.py`, `calibrate_marker.py` | **yes** | ground-truth accuracy against a running sim |
| corpus benchmarks *(next step)* | **no** | reproducible timing on any Arm device |

The corpus-based ones are the deliverable — they run on a Raspberry Pi, or on a
judge's machine, with no Gazebo anywhere. The live ones are how the corpus gets
trusted in the first place.

## Running the live tools

They need the gz bindings, which exist only inside the simulator container.
`in_sim.sh` handles the four things that are individually easy to get wrong:

```bash
bench/in_sim.sh bench/validate_live.py
bench/in_sim.sh bench/validate_live.py -n 300 --per-camera
bench/in_sim.sh bench/calibrate_marker.py --list-poses
```

What it takes care of, and why each one matters:

- **`-u abc`** — gz-transport discovery is per-user. Run as root and the camera
  topics simply are not there: no error, no frames, no explanation.
- **`HOME=/config`** — gz keeps discovery state under `HOME`. Without it,
  subscriptions never fire.
- **`/opt/mlvenv/bin/python`** — has gz, numpy, cv2 *and* yaml.
  `/usr/bin/python3` lacks yaml, so loading a site config fails on import.
- **`MSYS_NO_PATHCONV=1`** — Git Bash on Windows rewrites `/config/...` into a
  Windows path before docker sees it.

Prerequisite: the simulator's `docker-compose.yml` must mount this repository:

```yaml
volumes:
  - ../AI_Optimzation:/config/ai_opt
```

## `validate_live.py` — is the localization correct, and how fast?

Compares the pipeline's 3D estimate against simulator ground truth, and reports
per-stage timings. Fly the drone while it runs to measure under motion.

```
ERROR vs ground truth (mm)
  3D         mean     3.6  median     3.1  p95     6.3  max     6.7
  horizontal mean     3.2  median     2.9  p95     5.9  max     6.6
  vertical   mean    +0.9  sd    1.5

TIMING (ms per estimate)
  detect       mean   2.899
  triangulate  mean   0.343
  TOTAL        mean   3.253   -> 307.4 Hz ceiling (compute only)
```

Read the vertical error as **signed**. A large mean with a small spread means
`marker_dz` is wrong, not that the estimate is noisy — a distinction that costs
hours if you miss it. `--per-camera` shows the same thing at pixel level: a
consistent non-zero `dv` across *all* cameras is an offset error, and it is
invisible in reprojection residuals because every camera agrees on the wrong
answer.

`--occlude cam_ne,cam_sw` measures degradation with half the cameras gone.

## `calibrate_marker.py` — where is the detected point, relative to truth?

Run once per vehicle model. It exists because two things are easy to get wrong in
ways that produce plausible but incorrect numbers.

**Which pose is ground truth.** The pose topic publishes an entry per model *and*
per link. Link entries look authoritative and are static model-relative poses that
never move.

```bash
bench/in_sim.sh bench/calibrate_marker.py --list-poses     # fly while this runs
```

Only motion distinguishes a world-frame pose from a frozen one, so this is
inconclusive on a parked vehicle — and the tool says so rather than guessing.

**How high the detected point sits.** Reading it from the vehicle model file gives
the marker's height above its *parent link*, but truth is reported at the *model
origin*, which can sit lower. On the bundled x500 the model file says 0.18 m and
the measured offset is 0.4334 m — a 253 mm altitude bias that leaves horizontal
accuracy untouched, which is exactly why it survives casual checking.

```bash
bench/in_sim.sh bench/calibrate_marker.py
```

Samples above `--max-speed` (default 0.15 m/s) are discarded: motion blur drags
the detected centroid, biasing the offset low by ~40 mm during a climb versus a
1.4 mm spread while hovering. A static offset must be measured from static data.

Confirming it is *structural* needs two heights — a fixed offset and a
distance-dependent error are indistinguishable from one. Hover, measure, hover
somewhere else, measure again. Agreement to a millimetre or two settles it.
Measured on the bundled site: 0.4343 m on the ground, 0.4334 m at 2.5 m.
