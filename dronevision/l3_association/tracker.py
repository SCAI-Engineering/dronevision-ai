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
from collections import Counter

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
                 n_min=2, n_max=10, cam=None, crop=False, crop_pad=64):
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
        #: Search a `crop_pad`-px window around the Kalman prediction instead of
        #: the full frame, once a track exists. Costs a second, full-frame call
        #: whenever the window misses -- see `step`.
        self.crop = crop
        self.crop_pad = crop_pad
        self.crop_misses = 0             # window searched, target not in it

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

    def _window(self, img_rgb, cx, cy):
        """`crop_pad` px around (cx, cy), clipped to the frame. Returns (crop,
        (x0, y0)) so a detection inside it can be mapped back to full-frame
        pixels by adding the offset -- same convention `HybridDetector` uses
        for its motion-proposed crops."""
        h, w = img_rgb.shape[:2]
        x0 = max(0, int(cx - self.crop_pad))
        y0 = max(0, int(cy - self.crop_pad))
        x1 = min(w, int(cx + self.crop_pad))
        y1 = min(h, int(cy + self.crop_pad))
        return img_rgb[y0:y1, x0:x1], (x0, y0)

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

        # `kf.predict()` advances the filter's internal state, so it must run at
        # most ONCE per tick however this method exits. `px, py` holds that one
        # prediction and is scoped to the whole tick, because every path below
        # that would predict has to be able to see one already taken.
        px = py = None

        if due and self.det is not None:
            # Predicting BEFORE the search (rather than after, as the no-crop
            # path does) is what makes cropping possible: the window has to be
            # centred on where the target should be, which means knowing that
            # before looking, not after.
            search_img, (ox, oy) = img_rgb, (0, 0)
            if self.crop and self.have_track:
                px, py = self.kf.predict()
                search_img, (ox, oy) = self._window(img_rgb, px, py)

            det = self.det.detect(search_img, cam=self.cam)
            if det is None and (ox, oy) != (0, 0):
                # The window missed. Falling back to a full-frame search costs
                # a second inference on this tick -- strictly worse than the
                # no-crop path would have paid -- but the alternative is
                # reporting nothing on a track that has simply drifted, which
                # is worse than the extra cost.
                self.crop_misses += 1
                det = self.det.detect(img_rgb, cam=self.cam)
                ox, oy = 0, 0

            if det:
                x, y = float(det[0]) + ox, float(det[1]) + oy
                if self.have_track:
                    if px is None:
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
            # Reuse the prediction the crop path already took. Predicting again
            # here would advance the filter twice for one frame, so the next
            # window would be centred a frame too far ahead and the covariance
            # would grow at double rate -- and because it only happens when a
            # search failed, each miss would make the next one likelier. It also
            # only ever hit the cropping configurations, i.e. exactly the ones a
            # benchmark is measuring against the others.
            if px is None:
                px, py = self.kf.predict()
            x, y = px, py
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


class _CountingDetector:
    """Wraps a detector, counting every call regardless of outcome.

    `TrackedCamera.detect_rate` undercounts for benchmarking purposes: its
    `detections` counter only increments on a *successful* detect, so a call that
    ran the network and found nothing is invisible to it. That gap matters most
    exactly when it would be misleading -- a cluttered or fast-moving scene with
    frequent misses -- so cost accounting needs its own counter here rather than
    trusting the tracker's.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.pixels = 0     # sum of img.size over every call -- exact, not a
                            # "full frame every time" assumption. Matters once
                            # cropping is on: a window is smaller, a miss's
                            # full-frame retry is not, and only counting bytes
                            # actually handed to the network is honest either way.
        self.name = inner.name
        self.target_offset_key = getattr(inner, "target_offset_key", None)

    def detect(self, img, cam=None):
        self.calls += 1
        self.pixels += img.shape[0] * img.shape[1]
        return self.inner.detect(img, cam=cam)


class TrackedDetector:
    """Adapts per-camera `TrackedCamera` to the plain `Detector` interface
    (`detect(img, cam=None)`), so detect-then-track drops into
    `LocalizationPipeline`/`bench.accuracy` as a detector with no other change
    to either. One `TrackedCamera` per camera name, all sharing one underlying
    detector and one call counter.
    """

    def __init__(self, detector, cams, **tracker_kw):
        self._counted = _CountingDetector(detector)
        self.name = f"tracked/{detector.name}"
        self.target_offset_key = getattr(detector, "target_offset_key", None)
        self.modes = Counter()
        self._trackers = {c: TrackedCamera(detector=self._counted, cam=c, **tracker_kw)
                          for c in cams}

    @property
    def calls(self):
        """Real network invocations across every camera, success or not."""
        return self._counted.calls

    @property
    def pixels(self):
        """Total pixels handed to the network across every call."""
        return self._counted.pixels

    @property
    def crop_misses(self):
        """Crop windows that missed and paid for a full-frame retry, summed
        across every camera. Zero when `crop=False`."""
        return sum(t.crop_misses for t in self._trackers.values())

    def detect(self, img, cam=None):
        t = self._trackers.get(cam)
        if t is None:                      # camera this instance was not built for
            return self._counted.detect(img, cam=cam)
        x, y, mode = t.step(img)
        self.modes[mode] += 1
        return None if x is None else (x, y)
