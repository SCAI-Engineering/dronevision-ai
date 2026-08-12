# Decisions and lessons

## The findings that changed the implementation

### More processes can be slower

ONNX Runtime releases the Python GIL during inference, so threads already overlap. On the Pi 4, processes made the four-camera case 12% slower by copying 2.8 MB of frames per fix across process boundaries into an already memory-bound workload.

### A stale position is not a valid position

Frozen frames once produced a confident, stationary state after the camera service died. Per-camera timestamps and maximum-age checks now turn stale data into an explicit absence and recover automatically.

### Rejecting an outlier against its own corrupted solution fails

Single-pass reprojection rejection could blame the good cameras when one gross outlier pulled the shared solution far enough. Minimal-subset consensus replaced it, and the old failure remains pinned in a regression test.

### Ground truth must be audited

The simulator publishes both model and link poses. A link entry looked authoritative but was model-relative and static. The benchmark now identifies the moving model entry explicitly.

### The model file was not the effective calibration

The marker definition suggested a 0.18 m vertical offset. Measurement at multiple heights found a stable 0.4334 m effective offset. Using the file value created a 253 mm altitude bias while horizontal accuracy remained excellent—exactly the kind of plausible result that can survive superficial testing.

### JPEG can improve the system boundary

At quality 90, JPEG reduced bandwidth by roughly 70× and slightly improved the centroid of a tiny colour blob through mild low-pass filtering. Accuracy collapsed below quality 80, so compression remains a measured operating point rather than a blanket assumption.

### Export structure affects both deployment and accuracy

Removing the end-to-end selection head made the graph easier to delegate and improved localization from 333/337 to 337/337 on the FP32 raw reference. Export is part of model behaviour, not clerical conversion.

### Cropping does not make a fixed-shape network smaller

A Kalman-guided ROI experiment across 12 motion profiles removed most source pixels, but every crop was resized back to the model's fixed 320 × 320 tensor. Inference cost remained essentially unchanged. Dynamic shape or a genuinely smaller input graph is required for ROI to save neural compute.

### Frame skipping trades compute for motion error

Skipping detector frames raised throughput, but errors grew sharply during fast or nonlinear motion. Scheduling must therefore be evaluated against trajectories, not only hover throughput.

## Design decisions kept deliberately small

- No ROS dependency in the edge package.
- No Gazebo or PX4 import in the five AI layers.
- No full EKF duplicated ahead of PX4 EKF2.
- No pickle in the benchmark corpus.
- No silent CPU fallback when an NPU benchmark is requested.
- No performance claim without accuracy and artifact identity beside it.
