# The Arm optimization story

## Profile the full system first

The colour reference made the cost distribution unambiguous:

```text
acquire       0.007 ms
detect        1.924 ms  ← the layer worth optimizing
associate     0.003 ms
triangulate   0.300 ms
smooth        0.005 ms
```

Replacing colour segmentation with YOLO increases one row from milliseconds to hundreds of milliseconds on embedded CPUs. JPEG decode and geometry remain small. Every optimization therefore targets perception or the way camera inferences are scheduled.

## Experiment 1: spend four cores carefully

On the Raspberry Pi 4, four single-threaded camera workers reduce wall time, but each worker slows from 205 ms to 311 ms because all workers contend for a single-channel LPDDR4 memory system. Four workers deliver 2.58× throughput rather than the 4× suggested by core count.

The more surprising result is the process boundary:

| Pi 4 backend | Four-camera fix | Interpretation |
|---|---:|---|
| Threads | **319.9 ms** | ONNX Runtime releases the GIL, so inference overlaps |
| Processes | 358.7 ms | 2.8 MB of extra frame copies per fix worsen the memory bottleneck |

This changed the optimization order. More processes were not the flagship answer; fewer bytes moved became the priority.

## Experiment 2: compare Arm generations

The same FP32 ONNX artifact, input and runtime protocol reveal the platform gap:

| Board | Core configuration | Isolated camera inference | Throughput |
|---|---|---:|---:|
| Raspberry Pi 4 | Cortex-A72, 4 threads | 93.7 ms | 10.4 inf/s |
| Raspberry Pi 5 | Cortex-A76, 4 threads | 37.36 ms | 26.15 inf/s |
| NXP i.MX93 | Cortex-A55, 2 threads | 237.86 ms | 4.12 inf/s |

Pi 4 lacks Arm dot-product instructions; Pi 5 and i.MX93 expose `asimddp`. That matters for INT8 kernels, but the i.MX93's real advantage is its integrated Ethos-U65 rather than its two application CPU cores.

## Experiment 3: make the model accelerator-friendly

The original end-to-end ONNX graph carried selection operations that were poor NPU candidates. The export path was changed to expose a static raw head `[1, 5, 2100]`, leaving confidence filtering and NMS in the shared CPU codec.

That change unexpectedly improved the FP32 reference too:

| FP32 graph | Localized | Mean 3D | P95 3D |
|---|---:|---:|---:|
| End-to-end head | 333/337 | 36.42 mm | 77.58 mm |
| Raw head | **337/337** | **30.87 mm** | **59.13 mm** |

## Experiment 4: full-integer Ethos-U65

The deployed path is not merely weight-quantized. It uses INT8 input, INT8 activations and INT8 raw output, then compiles the graph with Vela for `ethos-u65-256`.

The final submission run reports:

- One node delegated out of one, with no CPU boundary nodes;
- 37.43 ms single-image inference;
- Approximately 6.35× speed-up over two-core i.MX93 FP32 inference;
- 336/337 localized sets;
- A model-size reduction from 9.31 MB to 2.42 MB, or 74%;
- 159.91 ms end-to-end, or 6.25 four-camera fixes/s.

The NPU path is **6.11× faster end-to-end than the two-core FP32 baseline** and fully delegated. Its 336/337 localization coverage shows that deployment performance and accuracy still need to be judged together.

## Experiment 5: test the attractive shortcuts

A Kalman-guided ROI experiment covered 12 motion profiles. Cropping reduced source pixels, but the fixed-shape ONNX graph still resized every crop to 320 × 320, so neural inference performed essentially the same work. **Fewer input pixels did not mean fewer network operations.**

Frame skipping did increase throughput, but fast and nonlinear motion produced large localization errors. Both are useful negative results: ROI inference needs a genuinely dynamic or smaller network shape, and adaptive scheduling needs motion-aware accuracy gates.

## Next levers

1. Export a dynamic-shape or genuinely smaller model for ROI inference.
2. Use quantization-aware training to recover post-training INT8 accuracy.
3. Improve NPU memory configuration and zero-copy/DMA data paths.
4. Expand motion datasets beyond hover and tune adaptive scheduling against them.
5. Demonstrate closed-loop point-to-point autonomous flight.
