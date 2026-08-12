# Why DroneVision

## The positioning problem moves off the aircraft

Indoor drones lose the most convenient source of position: GPS. Adding onboard cameras and compute increases mass, power draw and integration cost. DroneVision takes the opposite approach: **the environment observes the vehicle**, and an external Arm device computes the 3D position.

Four fixed cameras watch a 20 × 20 metre simulated industrial room. Each camera contributes a 2D observation. The AI block associates those views, triangulates a 3D point, rejects inconsistent cameras and publishes the estimate to downstream control software.

This produces a useful separation:

- The drone remains a normal PX4 vehicle;
- Camera and AI hardware can be serviced without touching the aircraft;
- A colour marker provides a fast, reliable safety baseline;
- A learned detector can be optimized and benchmarked without risking the flight loop.

## Why this is an Arm optimization project

The original design document identified the right target early: DLT, filtering and control are cheap; **per-camera learned perception is expensive**. A four-camera system multiplies every inference cost by four, so architectural differences that look small in a single-image benchmark become control-loop constraints.

The same YOLO26n artifact behaves very differently across the tested platforms:

| Platform | Relevant compute | What the project learns |
|---|---|---|
| Raspberry Pi 4 | 4 × Cortex-A72, no dot-product extension | More workers contend for memory; multiprocessing adds harmful copies |
| Raspberry Pi 5 | 4 × Cortex-A76 with dot product | Stronger cores improve FP32, but camera scheduling still matters |
| NXP i.MX93 FRDM | 2 × Cortex-A55 + Ethos-U65 | Full-integer export and graph delegation matter more than CPU threading |

The goal is not to report a single multiplier. It is to explain **which layer created it, what it cost in accuracy and whether it helps the complete four-camera fix**.

## Challenge fit

DroneVision was submitted to the **Physical AI track** of the Arm Create: AI Optimization Challenge 2026. The public challenge requirements ask projects to explain their overview, output, Arm setup and validation path, and encourage a clear demo. This site follows that structure while adding the engineering evidence behind the submission.

The deliverables are:

1. A reusable five-layer localization package;
2. A trained one-class YOLO26n detector and deployable artifacts;
3. A deterministic four-camera benchmark corpus;
4. Board-side automation and machine-readable measurements;
5. A companion Gazebo/PX4 simulator that demonstrates GPS-denied flight.

[:fontawesome-solid-trophy: View the Devpost submission](https://devpost.com/software/dronevision-ai){ .md-button .md-button--primary }
[:fontawesome-brands-github: Browse the AI repository](https://github.com/SCAI-Engineering/dronevision-ai){ .md-button }

## Success means more than FPS

A useful optimization must be evaluated at the **complete four-camera fix**, not only at single-image inference. It must also preserve localization coverage, report 3D error, keep input semantics unchanged and prove that the intended accelerator actually executed the graph.

That definition prevents three common shortcuts:

- Timing a single image while ignoring the other cameras;
- Claiming INT8 because weights are quantized while activations remain floating point;
- Reporting that a delegate loaded without checking CPU fallbacks.
