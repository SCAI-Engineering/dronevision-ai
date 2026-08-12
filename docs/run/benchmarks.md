# Benchmark toolkit

Everything under `bench/` answers a question with a number. It is deliberately outside the installed `dronevision` package.

## Primary tools

| Module | Question answered |
|---|---|
| `bench.accuracy` | How accurate is the complete 3D pipeline, and where is time spent? |
| `bench.speed` | How fast is one detector/runtime/thread configuration after warm-up? |
| `bench.parallel` | Should cores serve one inference or multiple cameras? |
| `bench.report` | How do recorded board results compare in one table? |
| `bench.record` | Can a live session become a deterministic replay corpus? |
| `bench.audit_geometry` | Is 3D error caused by projection matrices rather than detection noise? |
| `bench.refine_calibration` | Can bundle adjustment reduce structural reprojection bias? |
| `bench.quant_split` | Are calibration and holdout sets deterministic, complete and disjoint? |
| `bench.export_raw` | Can the detector expose an accelerator-friendly raw head? |
| `bench.export_tflite` | Does the export use the intended full-integer boundaries? |
| `bench.profile_ethosu_pmu` | What is the Ethos-U doing during inference? |

## Geometry audit chain

```text
audit_geometry  →  refine_calibration  →  accuracy
 detect drift       update geometry       verify 3D
```

High reprojection mean with low variance suggests a structural calibration bias. Low reprojection error shifts attention back to detector noise and multi-view geometry.

## Quantization data discipline

`data/corpus/quantization_split.json` chooses 64 evenly spaced frame sets across all four cameras: 256 calibration images. The remaining 273 sets, or 1,092 images, form the holdout.

The generator verifies:

- source ZIP and manifest hashes;
- camera completeness for every selected set;
- no overlap between calibration and holdout;
- complete corpus coverage;
- successful JPEG decode without re-encoding.

This makes calibration data part of the artifact provenance rather than an undocumented folder on one workstation.

## Read the JSON, not only the headline

Files in `bench/out/` retain model hashes, runtime versions, CPU features, thread counts, thermal state, stage distributions and worker-level timings. The documentation table is a view of that evidence, not the source of truth.

