"""2D constant-velocity Kalman filter, in pixel space. Pure numpy.

Used per camera by `dronevision.l3_association.tracker` to hold a 2D estimate alive
between the frames on which the detector actually ran. A predict step costs
microseconds against tens of milliseconds for a network inference, which is what
makes visiting one camera per tick a viable strategy rather than a degradation.
"""
import numpy as np


class KalmanCV:
    """State ``[x, y, vx, vy]``, measurement ``[x, y]``, time step of one frame.

    `q` is process noise (how much unmodelled acceleration to expect) and `r` is
    measurement noise (how much to distrust the detector). Their ratio sets how
    hard the filter pulls toward each new detection.
    """

    def __init__(self, q=4.0, r=6.0):
        self.x = None
        self.P = np.eye(4) * 500.0
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], float)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        #: Consecutive predicts without an update. The caller uses this to decide
        #: when a coasted track has gone stale enough to abandon.
        self.miss = 0

    @property
    def initialized(self):
        return self.x is not None

    def init(self, x, y):
        """Start a track at a measured position, with unknown velocity."""
        self.x = np.array([x, y, 0.0, 0.0], float)
        self.P = np.eye(4) * 500.0       # large: the initial velocity is a guess
        self.miss = 0

    def predict(self):
        """Advance one frame. Returns the predicted ``(x, y)``."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, x, y):
        """Fold in a measured position and clear the miss counter."""
        z = np.array([x, y], float)
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss = 0
        return self.x[0], self.x[1]

    @property
    def position(self):
        return None if self.x is None else (self.x[0], self.x[1])

    @property
    def velocity(self):
        """Pixels per frame."""
        return None if self.x is None else (self.x[2], self.x[3])
