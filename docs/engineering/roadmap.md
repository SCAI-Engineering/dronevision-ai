# Roadmap

## Proven now

- [x] Four-camera Gazebo/PX4 GPS-denied flight loop
- [x] Switchable colour, YOLO, motion and hybrid detectors
- [x] Five-layer portable AI package with enforced boundaries
- [x] Consensus multi-view triangulation and stale-frame handling
- [x] Deterministic 337-set corpus and benchmark toolkit
- [x] Raspberry Pi 4 and Pi 5 CPU measurements
- [x] NXP i.MX93 CPU validation and full test-suite pass
- [x] Raw-head model export and deterministic quantization split
- [x] Full-integer Vela artifact with complete Ethos-U65 delegation
- [x] End-to-end NPU accuracy and scheduling results

## Next engineering milestones

### Close the performance gap

- [ ] Sweep 320, 256 and 192 pixel inputs against 3D error
- [x] Test state-guided ROI cropping across 12 motion profiles; fixed 320 × 320 export showed no inference saving
- [x] Test frame skipping; throughput improved but fast/nonlinear motion error was unacceptable
- [ ] Export a dynamic-shape or smaller-input model that can make ROI inference cheaper
- [ ] Add accuracy-gated, motion-aware inference cadence and detector scheduling
- [ ] Explore double buffering around the single NPU queue
- [ ] Improve NPU memory configuration and DMA/zero-copy data paths
- [ ] Use quantization-aware training to recover INT8 accuracy

### Broaden evidence

- [ ] Record varied flight trajectories rather than hover only
- [ ] Separate quantization holdout from the final accuracy corpus
- [ ] Run 10–15 minute sustained thermal tests
- [ ] Measure live network-to-state latency and jitter
- [ ] Validate against real USB or IP cameras

### Move toward deployment

- [ ] Define explicit loss-of-tracking and landing safety policy
- [ ] Calibrate a physical multi-camera room
- [ ] Harden board networking and service supervision
- [ ] Evaluate multi-room handoff and multi-target identity

## The current honest status

The classical detector is fast and precise on i.MX93, but it needs a marker. The final submitted learned NPU path is fully delegated, reaches 6.25 fixes/s and reduces the complete-loop latency by 6.11× versus the two-core FP32 baseline. The next milestone is not another isolated inference number; it is a learned path that improves speed while recovering quantized accuracy across representative motion and supports closed-loop point-to-point flight.
