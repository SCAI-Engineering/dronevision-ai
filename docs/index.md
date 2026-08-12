---
title: DroneVision AI
description: A camera-based positioning system that lets a drone fly indoors without GPS or onboard perception.
hide:
  - navigation
  - toc
---

<div class="dv-hero">
  <div class="dv-hero-copy">
    <p class="dv-kicker">External vision · Edge AI · Arm</p>
    <h1>The room becomes<br><span>the drone's eyes.</span></h1>
    <p class="dv-lede">DroneVision turns four fixed camera feeds into a live 3D position that PX4 can fly on—without GPS and without putting perception hardware on the aircraft.</p>
    <div class="dv-actions">
      <a class="md-button md-button--primary" href="project/why/">Explore the system</a>
      <a class="md-button" href="run/reproducibility/">Reproduce the results</a>
    </div>
  </div>
  <div class="dv-field" aria-label="Four cameras triangulating a drone position">
    <span class="dv-grid"></span>
    <span class="dv-camera dv-camera--nw"><i></i><b>NW</b></span>
    <span class="dv-camera dv-camera--ne"><i></i><b>NE</b></span>
    <span class="dv-camera dv-camera--sw"><i></i><b>SW</b></span>
    <span class="dv-camera dv-camera--se"><i></i><b>SE</b></span>
    <span class="dv-ray dv-ray--a"></span>
    <span class="dv-ray dv-ray--b"></span>
    <span class="dv-crosshair"><i></i></span>
    <span class="dv-lock">3D LOCK <b>6.28 mm</b></span>
  </div>
</div>

<div class="dv-statusbar" role="list" aria-label="Project highlights">
  <div role="listitem"><strong>4</strong><span>synchronized views</span></div>
  <div role="listitem"><strong>337</strong><span>reproducible frame sets</span></div>
  <div role="listitem"><strong>99.70%</strong><span>NPU localization success</span></div>
  <div role="listitem"><strong>191</strong><span>hardware-free tests</span></div>
</div>

## A real control workload—not a microbenchmark

DroneVision was built for the **Physical AI track of the Arm Create: AI Optimization Challenge 2026**. It combines a complete GPS-denied flight loop with a measurement framework that explains where the time goes on real Arm boards.

<div class="dv-card-grid dv-reveal">
  <article class="dv-card">
    <span class="dv-card-label">Observe</span>
    <h3>Four fixed cameras</h3>
    <p>JPEG frames arrive as synchronized sets. Stale views are rejected instead of silently freezing the output.</p>
  </article>
  <article class="dv-card">
    <span class="dv-card-label">Understand</span>
    <h3>Switchable perception</h3>
    <p>A reliable colour baseline and a learned YOLO26n detector share the same geometry and state-estimation pipeline.</p>
  </article>
  <article class="dv-card">
    <span class="dv-card-label">Locate</span>
    <h3>Consensus triangulation</h3>
    <p>Multi-view DLT rejects gross outliers, degrades gracefully with occlusion and emits an honest 3D estimate.</p>
  </article>
  <article class="dv-card">
    <span class="dv-card-label">Fly</span>
    <h3>PX4 external vision</h3>
    <p>The companion simulator injects the estimate into EKF2 as external vision and keeps the X500 airborne with GPS disabled.</p>
  </article>
</div>

## The optimization result

The profiler made the decision for us: **detection dominates; geometry does not**. On the NXP i.MX93, the final submitted Ethos-U65 path reduced the full four-camera YOLO loop from **977.7 ms on FP32 CPU** to **159.91 ms per fix**—a measured **6.11× end-to-end speed-up**. The accelerated path localizes 336 of 337 frame sets, while the model shrinks from 9.31 MB to 2.42 MB.

<div class="dv-result-band dv-reveal">
  <div>
    <span>FP32 / Cortex-A55</span>
    <strong>1.02 <small>fixes/s</small></strong>
    <i style="--bar: 17%"></i>
  </div>
  <div class="is-accent">
    <span>INT8 / Ethos-U65</span>
    <strong>6.25 <small>fixes/s</small></strong>
    <i style="--bar: 99%"></i>
  </div>
</div>

[Read the optimization story →](optimization/index.md){ .dv-text-link }

## One codebase, three kinds of proof

| Proof | What ships | Why it matters |
|---|---|---|
| **Flight proof** | Gazebo warehouse, PX4 X500, four cameras and GPS-denied control | Shows that the estimate is useful, not merely numerically plausible |
| **Accuracy proof** | A 337-set corpus with time-aligned ground truth | Makes every detector and runtime comparison repeatable |
| **Hardware proof** | Pi 4, Pi 5 and i.MX93 measurements with artifact hashes | Attributes speed-ups to the architecture and runtime actually used |

!!! note "Current state"
    The system is functional and measured. After auditing how the marker offset interacts with bundle adjustment, the current nominal factory geometry reproduces 337/337 sets at 6.28 mm mean error. The learned i.MX93 NPU path reaches 6.25 four-camera fixes/s with 100% graph delegation. The hover corpus does not yet establish performance across varied real rooms or closed-loop point-to-point flight. These limits are documented, not hidden.

<div class="dv-next">
  <p>Start with the system story, then inspect every number.</p>
  <a class="md-button md-button--primary" href="project/architecture/">Open the architecture</a>
  <a class="md-button" href="optimization/results/">See measured results</a>
</div>
