"""Layer 4 — camera geometry and multi-view triangulation.

Turns 2D pixel detections from two or more cameras into one 3D point, by DLT
(direct linear transform, solved with an SVD), with outlier rejection: the
candidate point is reprojected into every contributing camera and views that
disagree beyond a pixel threshold are dropped before re-solving. That is what
lets the system keep working when a camera is occluded.

Camera intrinsics and extrinsics are loaded from a site configuration file
(`config/<site>.yaml`, see `calibration.py`) rather than hard-coded, so a
different room — or real cameras with a real calibration — is a config change and
not a code change.

This layer is cheap: well under a millisecond for four views. It is not an
optimization target, and it is deliberately left alone.
"""
