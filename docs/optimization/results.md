# Results and methodology

## Accuracy reference

All accuracy rows use the committed 337-set corpus and synchronized ground truth. The
intermediate rows preserve the detailed validation snapshots recorded during development;
the refined classical result and final submitted run are the latest project figures.

| Pipeline | Localized | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Colour marker, before geometry refinement | 337/337 | 6.22 mm | 6.02 mm | 10.72 mm | 14.79 mm |
| **Colour marker, refined calibration** | **337/337** | **5.02 mm** | **4.95 mm** | **8.55 mm** | — |
| YOLO FP32, end-to-end | 333/337 | 36.42 mm | 32.07 mm | 77.58 mm | 181.64 mm |
| YOLO FP32, raw head | **337/337** | 30.87 mm | 28.10 mm | 59.13 mm | 162.78 mm |
| YOLO INT8, Ethos-U65 | 336/337 | 52.39 mm | — | 95.94 mm | — |

The colour result is a reference, not an AI target: it depends on a visible marker. The learned detector answers the harder deployment question.

## Four-camera performance

### NXP i.MX93 FRDM

| Configuration | Mean total | Frame sets/s | Mean 3D error |
|---|---:|---:|---:|
| Colour, decoded (earlier validation snapshot) | 30.23 ms | 33.09 | 6.22 mm |
| YOLO FP32 CPU, 1 thread | 1,607.16 ms | 0.62 | 36.42 mm |
| YOLO FP32 CPU, 2 threads | 977.73 ms | 1.02 | 36.42 mm |
| YOLO FP32 CPU, 2 threads + JPEG | 990.76 ms | 1.01 | 36.42 mm |
| YOLO INT8 Ethos-U65, sequential | 199.37 ms | 5.02 | 52.39 mm |
| **YOLO INT8 Ethos-U65, 2 workers** | **167.78 ms** | **5.96** | **52.39 mm** |
| **Final submitted Ethos-U65 path** | **159.91 ms** | **6.25** | See submission accuracy record |

The first two NPU rows are preserved from the detailed validation snapshot; the final row
is the later submitted run. JPEG adds approximately 13 ms per set—only 1.33% of the FP32
total. The dominant cost is inference.

### Raspberry Pi 5 scheduling

Giving all four cores to one inference is marginally better than splitting them across four simultaneous cameras:

| Strategy | Four-camera fix | Fixes/s |
|---|---:|---:|
| 1 worker × 4 threads | 115.89 ms | 8.63 |
| **2 workers × 2 threads** | **114.50 ms** | **8.73** |
| 4 workers × 1 thread | 120.99 ms | 8.27 |

This is close enough that thermal state and run ordering matter. The repository stores temperatures, clock rates, pass spread and per-worker inference time alongside the headline numbers.

## Measurement protocol

Every cross-board isolated result uses:

- `drone_yolo26n_v4.onnx` with SHA prefix `7acd721e718a`;
- static 320 × 320 input and batch size 1;
- explicit runtime and thread counts;
- warm-up iterations before sampling;
- an otherwise idle board where possible;
- mean, median, P95, minimum, maximum and standard deviation;
- thermal and CPU-feature metadata in JSON.

Accuracy is deterministic on the fixed bytes. Timing is not. Comparisons therefore belong inside a controlled sweep rather than across unrelated machine states.

## Evaluation dimensions

| Dimension | Reference | Current learned path |
|---|---:|---:|
| Localization coverage | 337/337 raw FP32 | 336/337 INT8 NPU |
| Mean 3D error | 30.87 mm raw FP32 | 52.39 mm in the detailed NPU validation run |
| P95 3D error | 59.13 mm raw FP32 | 95.94 mm in the detailed NPU validation run |
| Complete four-camera fix | 977.73 ms FP32 CPU | 159.91 ms final submitted NPU run |

There is no arbitrary fixed-rate pass/fail threshold. Results are reported as a speed–accuracy profile: complete-loop latency, localization coverage and 3D error remain visible together.

!!! warning "What these results do not prove"
    The primary corpus contains a mostly hovering drone. It does not establish accuracy for fast trajectories, varied real rooms, motion-only detection, long thermal runs or live network jitter. Those are explicit roadmap items.
