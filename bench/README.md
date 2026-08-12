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

`python -m bench.export_raw` exports the trained YOLO26 checkpoint with `end2end=False`.
The resulting static FP32 ONNX graph emits decoded boxes and class scores as
`[1,5,2100]`; detection selection stays in the shared codec so that later INT8/TFLite
exports do not carry `TopK` or NMS into the accelerator graph. The script pins and records
the original Ultralytics exporter version and source checkpoint hash.

Two kinds, and the difference matters:

| Mode | Source | Purpose |
|---|---|---|
| **Live** | TCP Endpoint (`--frames`) | Real-time accuracy against a running sim/hardware |
| **Corpus** | Recorded Data (default) | Reproducible timing and precision on any device |

The corpus benchmarks are the primary deliverable — they run on a Raspberry Pi, or on a
judge's machine, with no simulator required. Live measurements are used to verify 
the system in real-time or to generate new corpora.

## `accuracy.py` — The Primary Benchmark Tool

This is the "Swiss Army Knife" for measuring both precision and performance. It compares 
the pipeline's 3D estimate against ground truth and reports per-stage timings.

### Measuring Accuracy
Run against the recorded corpus:
```bash
python -m bench.accuracy
```
Or measure live from a simulator/camera service:
```bash
python -m bench.accuracy --frames tcp://127.0.0.1:5555 -n 200
```

### Calibrating the Marker Offset
If you see a large vertical mean error with a small spread, your `marker_dz` is likely wrong. Use `--marker-offset` to measure the detected point's offset from truth directly:
```bash
python -m bench.accuracy --marker-offset
```

### Performance Analysis
The tool reports timing for each stage (acquire, detect, associate, triangulate, smooth). 
Use this to identify bottlenecks before optimizing specific layers.

## Geometry Audit & Calibration Chain

When 3D accuracy is poor but detector noise seems low, the problem is usually structural drift in the Projection Matrices ($P$). Use these tools to isolate and fix it:

### `audit_geometry.py` — Reprojection Error Analysis
Measures reprojection error in pixels using ground-truth 3D points. This isolates $P$ matrix inaccuracy from detector noise.
- **High Mean / Low StdDev:** Indicates structural bias (calibration drift).
- **Low Mean:** Geometry is sound; any remaining 3D error is due to detector noise.

```bash
python -m bench.audit_geometry --site factory --detector color
```

### `refine_calibration.py` — Bundle Adjustment Refinement
Optimizes camera extrinsics (positions and rotations) using a non-linear least-squares approach (Bundle Adjustment) to minimize reprojection error. It outputs suggested updates for the site YAML config.

```bash
python -m bench.refine_calibration --site factory --detector color
```

**Important:** When refining calibration with detectors that have a known offset from ground truth (e.g., color marker detector), use the `--marker-offset` flag to account for this offset during optimization. Without it, the bundle adjustment will absorb the offset into camera extrinsics, improving reprojection error while degrading 3D triangulation accuracy.

```bash
python -m bench.refine_calibration --site factory --detector color --corpus data/corpus_updown_smooth --marker-offset
```

**Workflow:** `audit_geometry` (Detect Drift) $\rightarrow$ `refine_calibration` (Fix $P$) $\rightarrow$ `accuracy` (Verify 3D).

## Other Tools

- `speed.py`: Focused performance sweep across different runtimes/threads.
- `report.py`: Renders the results of speed sweeps into a readable table.
- `parallel.py`: Measures the efficiency of multi-core detection strategies.
- `record.py`: Captures live data to create a new corpus for reproducible testing.
1.4 mm spread while hovering. A static offset must be measured from static data.

Confirming it is *structural* needs two heights — a fixed offset and a
distance-dependent error are indistinguishable from one. Hover, measure, hover
somewhere else, measure again. Agreement to a millimetre or two settles it.
Measured on the bundled site: 0.4343 m on the ground, 0.4334 m at 2.5 m.
