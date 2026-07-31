"""Per-camera temporal association: detect on some frames, track on the rest.

One instance owns one camera. The detector runs every Nth frame; in between, a
constant-velocity Kalman filter carries the estimate forward. On an Arm CPU where
a single inference costs tens of milliseconds and a Kalman predict costs
microseconds, this is the difference between a pipeline that closes a control loop
and one that does not.

Three levers, in descending order of what they actually buy:

  detect_every    run the network on 1 frame in N. The main lever.
  adaptive        vary N by how well the filter is predicting. When the target
                  moves predictably, look less often; when it does not, look more.
  use_gate        skip the network entirely when nothing is moving. Off by
                  default: at the reduced resolution the gate runs at, a small
                  distant target does not reliably register as motion, so the gate
                  discards real detections. It pays off only where the target is
                  usually absent, not merely still.
"""
import math

import cv2
import numpy as np

from dronevision.l3_association.kalman2d import KalmanCV


class TrackedCamera:
    """Detect-then-track for a single camera.

    Frames are RGB, matching every detector and every frame source in this
    package. `step()` returns ``(x, y, mode)`` with mode one of:

        detect   the network ran and found the target
        track    coasting on the Kalman prediction
        gated    motion gate saw nothing, so the network was skipped
        idle     no track, and nothing found
    """

    def __init__(self, detector=None, detect_every=5, min_motion_area=25,
                 max_miss=15, mog_width=320, use_gate=False, adaptive=False,
                 n_min=2, n_max=10, cam=None):
        self.det = detector
        self.cam = cam
        self.every = detect_every
        self.min_area = min_motion_area
        self.max_miss = max_miss
        self.mog_width = mog_width       # gate at reduced width; full res costs as
                                         # much as the inference it is meant to save
        self.use_gate = use_gate
        self.adaptive = adaptive
        self.n_min = n_min
        self.n_max = n_max
        self.n_cur = detect_every
        self.since = math.inf            # force a detection on the first frame
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=30, detectShadows=False)
        self.kf = KalmanCV()
        self.have_track = False
        self.frames = 0
        self.detections = 0

    @property
    def detect_rate(self):
        """Fraction of frames on which the network actually ran."""
        return self.detections / self.frames if self.frames else 0.0

    def _motion(self, img_rgb):
        """Contours of moving regions, computed at reduced resolution."""
        h, w = img_rgb.shape[:2]
        scale = self.mog_width / w
        small = cv2.resize(img_rgb, (self.mog_width, max(1, int(h * scale))))
        fg = self.mog.apply(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= self.min_area]

    def _adapt(self, residual):
        """Tighten or relax the detection interval by prediction error."""
        if residual > 20:                        # prediction is poor: look more often
            self.n_cur = max(self.n_min, self.n_cur - 2)
        elif residual < 8:                       # prediction is good: look less often
            self.n_cur = min(self.n_max, self.n_cur + 1)

    def step(self, img_rgb, force_detect=False):
        """Advance one frame. `force_detect` overrides the schedule."""
        self.frames += 1
        self.since += 1

        if self.use_gate and not self.have_track and not force_detect:
            if not self._motion(img_rgb):
                return None, None, "gated"

        n = self.n_cur if self.adaptive else self.every
        due = (force_detect or self.since >= n or not self.have_track
               or self.kf.miss >= self.max_miss)

        if due and self.det is not None:
            det = self.det.detect(img_rgb, cam=self.cam)
            if det:
                x, y = float(det[0]), float(det[1])
                if self.have_track:
                    px, py = self.kf.predict()
                    if self.adaptive:
                        self._adapt(math.hypot(x - px, y - py))
                    self.kf.update(x, y)
                else:
                    self.kf.init(x, y)
                    self.have_track = True
                self.since = 0
                self.detections += 1
                return x, y, "detect"

        if self.have_track:
            x, y = self.kf.predict()
            self.kf.miss += 1
            if self.kf.miss > self.max_miss:
                # The track has coasted too long to be believable. Drop it rather
                # than keep feeding triangulation an increasingly fictional point.
                self.have_track = False
                self.kf.x = None
                return None, None, "idle"
            return x, y, "track"

        return None, None, "idle"

    def reset(self):
        self.have_track = False
        self.kf = KalmanCV()
        self.since = math.inf


#: The class was previously named for the board it was written for. Kept so
#: existing benchmark scripts continue to import successfully.
RpiPipeline = TrackedCamera
