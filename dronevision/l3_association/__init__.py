"""Layer 3 — 2D Detection Association.

Two distinct jobs, easy to confuse:

  * across cameras — decide which detections belong to the same physical target.
    With a single drone in the scene this is trivial (camera identity is target
    identity); the seam exists for the multi-target case.
  * across time, per camera — keep a camera's 2D estimate alive between the frames
    on which the detector actually ran, using a constant-velocity Kalman filter to
    coast.

The second job is what makes a CPU-bound Arm deployment viable. If the detector
visits only one camera per tick, the other cameras still need a current 2D position
for triangulation to work — coasting supplies it, at a cost of roughly nothing
compared with a network inference.

`kalman2d.py` is that filter, and it lives here rather than with the 3D estimators
in layer 5: coasting a camera's pixel estimate between detections is temporal
association, this is its only consumer, and filing it downstream made layer 3
depend on a later layer.
"""
