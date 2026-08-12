# Motion trajectories for corpus recording

## Executive summary

The corpora behind every benchmark in this repository — including the [ROI benchmark](hw-comparison-pi5-imx93.md) — are not hand-flown. They come from `teleport_path.py`, a script in the companion simulator repository that writes a Gazebo model pose directly, on a fixed clock, from one of seven closed-form motion generators. Commanding the pose instead of flying it makes every recording exactly reproducible from a one-line CLI invocation and lets a motion run at any speed, sim-time or otherwise, without a flight controller in the loop. Twelve corpora, each a single motion generator at a specific parameterization, form the full trajectory set used across the accuracy and throughput benchmarks; this page documents the seven generators and the twelve recordings drawn from them.

## Provenance

| Item | Value |
|---|---|
| Script | `tools/dev/teleport_path.py`, repository `Hackathon_ARM_Simulator` |
| Commit | `13a1957` — "Add scripted motion generator and trajectory console", merged to `dev` |
| Interactive preview | `tools/dev/trajectory_console.html` — same directory, no build step, no server, pure JS reimplementation of the same formulas |
| Consumer | `bench.record` in this repository, and every corpus under `data/corpus_*` used by [`bench.accuracy`](run/benchmarks.md) |

## Why scripted teleport, not flown motion

Flying each trajectory by hand through PX4 would make every recording depend on stick input that cannot be exactly repeated, and would tie recording speed to real flight dynamics rather than to whatever probe speed a given test actually needs. `teleport_path.py` instead computes a closed-form pose — position, and for the attitude-exercising motions, orientation — at a fixed rate against the **simulation clock**, not the wall clock, and writes it straight to the model state. This keeps a `--duration 12` run at exactly 12 simulated seconds regardless of real-time factor, and makes every corpus reproducible byte-for-byte from the command line alone.

!!! warning "Kill the previous process before every run"
    `Ctrl-C` from outside the simulator container does not reliably stop `teleport_path.py` running inside it — `docker exec` without a TTY often fails to forward the signal, leaving an orphaned process still pushing poses at 100 Hz. Running a second motion on top of it produces two processes fighting over the same probe: ground truth position jumps incoherently and the speed readout spikes to nonsense. It looks exactly like a simulator bug and isn't one. Always `pkill -f teleport_path.py` before starting the next motion, including the first one of a session.

## The seven motions

| Motion | Exercises | Key parameters (defaults) |
|---|---|---|
| `hover` | Near-static baseline, optional small positional jitter | `--alt` 2.5 m, `--yaw` 0°, `--jitter` 0 m, `--jitter-hz` 0.5 |
| `pitch` | Oscillating pitch/roll attitude at a fixed position | `--pitch-amp` 15°, `--pitch-hz` 0.2, `--roll-amp` 0°, `--roll-hz` 0.2 |
| `spin` | Continuous body-rate rotation about roll/pitch/yaw | `--roll-rate` 60°/s, `--pitch-rate` 0°/s, `--yaw-rate` 0°/s |
| `linear` | Constant-speed translation between two points | `--from`/`--to` (required), `--speed` 2.0 m/s, `--mode` bounce |
| `updown` | Vertical oscillation between two altitudes | `--alt-low` 1.0 m, `--alt-high` 4.0 m, `--period` 4.0 s/leg, `--profile` smooth |
| `ramp` | Translation with a velocity ramp instead of constant speed | `--from`/`--to` (required), `--v0` 0.5 m/s, `--v1` 3.0 m/s, `--ramp-time` 3.0 s, `--profile` smooth |
| `square` | Closed rectangular path at constant speed and altitude | `--side` 6.0 m, `--speed` 2.0 m/s |

All seven also accept `--rate` (100 Hz default, teleport update rate), `--duration` (0 = run until interrupted, in simulated seconds), `--world` (`factory`), `--model` (`probe`), `--vehicle` (`x500_0`), and a discouraged `--no-probe` flag that teleports the real vehicle entity directly and can crash it out of the simulation.

!!! tip "See a parameter change before running it"
    `tools/dev/trajectory_console.html` is an interactive, dependency-free page — open it straight from disk, no server or build step — that plots each motion live as its parameters are edited. It is a hand-ported reimplementation of the exact same formulas as `teleport_path.py`, checked against the Python output, so dragging `--pitch-amp` or `--period` and watching the curve change is a faster way to build intuition for what a parameter does than reading the table above, before spending simulated time recording it for real.

## Running a motion

Each command below is the bare CLI form. Inside the simulator container it runs as `HOME=/config /opt/mlvenv/bin/python teleport_path.py <args>` from `tools/dev/`, wrapped in `docker exec -u abc gazebo-web bash -lc '...'` against the container defined in `docker-compose.yml`; the console at `tools/dev/trajectory_console.html` generates that exact wrapped form for any parameter combination and is the fastest way to preview a motion before recording it.

=== "Hover"
    ```bash
    python teleport_path.py hover --alt 2.5 --jitter 0.15 --jitter-hz 0.5
    ```

=== "Pitch"
    ```bash
    python teleport_path.py pitch --pitch-amp 15 --pitch-hz 0.2
    ```

=== "Spin"
    ```bash
    python teleport_path.py spin --roll-rate 60
    ```

=== "Linear"
    ```bash
    python teleport_path.py linear --from -3 -3 --to 3 3 --speed 2.0 --mode bounce
    ```

=== "Up/Down"
    ```bash
    python teleport_path.py updown --alt-low 1.0 --alt-high 4.0 --period 4.0 --profile smooth
    ```

=== "Ramp"
    ```bash
    python teleport_path.py ramp --from -3 0 --to 3 0 --v0 0.5 --v1 3.0 --ramp-time 3.0
    ```

=== "Square"
    ```bash
    python teleport_path.py square --side 6.0 --speed 2.0
    ```

Before every invocation, the control room needs to be running with `TRUTH_VEHICLE=probe` so it tracks the teleported probe rather than the parked vehicle, and any previous `teleport_path.py` process needs to be killed first, per the warning above.

## The twelve recorded corpora

Twelve corpora under `data/corpus_*` back every benchmark in this repository, each one motion generator at a specific parameterization chosen to stress a different regime — slow vs. fast translation, smooth vs. stepped profiles, roll-dominant vs. yaw-dominant rotation:

| Corpus | Motion | Regime |
|---|---|---|
| `hover_jitter` | `hover` | Near-static baseline with small positional jitter |
| `pitch` | `pitch` | Oscillating attitude, fixed position |
| `spin_roll` | `spin` | Continuous roll-dominant rotation |
| `spin_yaw` | `spin` | Continuous yaw-dominant rotation |
| `linear_slow` | `linear` | Low-speed constant translation |
| `linear_fast` | `linear` | High-speed constant translation |
| `updown_smooth` | `updown` | Smooth vertical oscillation |
| `updown_step` | `updown` | Stepped vertical oscillation |
| `ramp_smooth` | `ramp` | Smooth velocity ramp |
| `ramp_step` | `ramp` | Stepped velocity ramp |
| `square` | `square` | Closed rectangular path, moderate speed |
| `square_fast` | `square` | Closed rectangular path, high speed |

This is the full trajectory set reported in the [ROI benchmark](hw-comparison-pi5-imx93.md) across the Raspberry Pi 5, i.MX93 CPU, and i.MX93 Ethos-U65 NPU, and in the accuracy tables in [Results & methodology](optimization/results.md).
