# Hackathon submission

## DroneVision AI

**Arm Create: AI Optimization Challenge 2026 · Physical AI track**

DroneVision is an external camera-based positioning system for indoor drones. Four fixed cameras detect the vehicle, triangulate its 3D position and provide external vision to PX4 so the aircraft can fly without GPS or onboard perception hardware.

The project contributes both a working Physical AI application and an attributable Arm optimization study. It demonstrates GPS-denied flight in Gazebo/PX4, ships a deterministic benchmark corpus, and measures the same YOLO26n workload on Raspberry Pi 4, Raspberry Pi 5 and NXP i.MX93. The final submitted embedded result uses a fully integer, fully delegated Ethos-U65 graph, reaches 37.43 ms single-image inference and processes a complete four-camera fix in 159.91 ms—6.11× faster end-to-end than the two-core i.MX93 FP32 baseline.

## Submission highlights

- Refined classical reference: 5.02 mm mean, 4.95 mm median and 8.55 mm P95 3D error across 337 synchronized frame sets.
- Ethos-U65: 100% neural graph delegation with no CPU fallback islands.
- Final localization rate: 6.25 complete four-camera fixes/s, with 336/337 successful frame sets.
- Model size: 9.31 MB to 2.42 MB, a 74% reduction.
- Architecture study: Pi 4 memory-bandwidth limits, Pi 5 scheduling behaviour and i.MX93 NPU acceleration.
- Negative results: multiprocessing copies, fixed-shape ROI cropping and motion-sensitive frame skipping.

## What comes next

The next work focuses on native end-to-end INT8 boundaries, NPU memory configuration, DMA/zero-copy paths, better multi-camera pipelining, quantization-aware training, dynamic-shape ROI inference, motion-rich datasets and closed-loop point-to-point autonomous flight.

## Submission links

- [Devpost project page](https://devpost.com/software/dronevision-ai)
- [DroneVision AI source code](https://github.com/SCAI-Engineering/dronevision-ai)
- [Gazebo/PX4 companion simulator](https://github.com/SCAI-Engineering/Hackathon_ARM_Simulator)
- [Official challenge rules](https://arm-ai-optimization-challenge.devpost.com/rules)

## Built with

Python · OpenCV · YOLO26n · ONNX Runtime · TensorFlow Lite · Arm Vela · Ethos-U65 · NumPy · ZeroMQ · Gazebo Harmonic · PX4 SITL · Docker

## License

The DroneVision AI repository is released under the Apache License 2.0. Third-party model, runtime and simulator components remain subject to their respective licenses.
