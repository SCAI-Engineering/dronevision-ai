"""Constant-velocity 3D EKF — SCAFFOLD, not wired into any running path.

State, covariance and measurement model are defined; `predict` and `update` raise.
See `dronevision.l5_estimation` for why an EMA is what actually runs: the flight
controller downstream performs the real fusion against inertial data, so a second
full estimator here would duplicate it rather than improve it.

The scaffold is kept because the interface is the useful part. If this ever gets
implemented, the case for it is not smoother output — it is that a predicted state
lets each camera's detection be fused as an independent bearing measurement at its
own timestamp, instead of requiring several cameras to agree on one instant before
any 3D point can be produced at all.
"""
import numpy as np


class EKF3D:
    """State ``x = [x, y, z, vx, vy, vz]``, measuring position only."""

    def __init__(self, q=1.0, r=0.05):
        self.x = None
        self.P = np.eye(6) * 10.0
        self.q = float(q)                                  # process noise density
        self.r = float(r)                                  # measurement noise (m)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])   # observe position only

    def init(self, xyz):
        self.x = np.array([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0], float)
        self.P = np.eye(6) * 10.0

    def predict(self, dt):
        raise NotImplementedError("EKF3D is a scaffold; see the module docstring")

    def update(self, z_xyz, R=None):
        raise NotImplementedError("EKF3D is a scaffold; see the module docstring")

    @property
    def position(self):
        return None if self.x is None else self.x[:3]

    @property
    def velocity(self):
        return None if self.x is None else self.x[3:]
