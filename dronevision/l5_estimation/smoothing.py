"""Temporal smoothing of the triangulated 3D position.

A single-pole exponential moving average — position only, no velocity state, no
covariance. See `dronevision.l5_estimation` for why that is a deliberate choice and
not a placeholder for the EKF in `ekf3d.py`.
"""
import numpy as np


class EMASmoother:
    """Exponential moving average over an N-dimensional point.

    `alpha` in [0, 1) weights the running estimate against each new measurement:
    0 passes measurements through untouched, values near 1 are heavily smoothed
    and heavily lagged. That lag is real and lands in a control loop, so it is
    reported alongside the estimate rather than hidden.
    """

    def __init__(self, alpha=0.5):
        alpha = float(alpha)
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.alpha = alpha
        self._state = None

    def update(self, x):
        """Feed a measurement, return the smoothed estimate as an ndarray."""
        x = np.asarray(x, float)
        self._state = x.copy() if self._state is None else (
            self.alpha * self._state + (1.0 - self.alpha) * x)
        return self._state

    @property
    def value(self):
        return self._state

    @property
    def initialized(self):
        return self._state is not None

    def reset(self):
        """Forget history. Call after a track loss, so a stale position cannot
        drag the first estimate of the new track toward where the target was."""
        self._state = None
