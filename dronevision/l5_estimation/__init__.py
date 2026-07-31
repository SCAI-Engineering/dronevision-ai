"""Layer 5 — state estimation and filtering.

What runs today is an exponential moving average over the triangulated position.
A constant-velocity 3D EKF is scaffolded (state, covariance, measurement model)
but is NOT wired into any live path — see `ekf3d.py`.

That is a deliberate choice rather than an omission. The flight controller
downstream runs its own EKF and performs the real fusion of this estimate with
inertial data; a second full EKF here would duplicate it. What this layer owes
the controller is a smooth, outlier-free position and an honest statement of how
stale it is — not a second opinion on velocity.

The 2D per-camera Kalman filter is NOT here. It coasts a single camera's pixel
estimate between detections, which is temporal association — layer 3's job — and
filing it here made layer 3 depend on a later layer.
"""
