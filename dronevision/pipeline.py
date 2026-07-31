"""End-to-end localization: frames in, one 3D position out.

Chains layers 2 through 5 — detect per camera, associate, triangulate, smooth —
behind a single object. Frame acquisition (layer 1) and the output sink are
injected by the caller, so the core carries no transport dependency and can be
driven from a recorded corpus as readily as from a live camera feed.

    pipeline = LocalizationPipeline(cal=load_site("factory"))
    est = pipeline.locate(frames)        # {cam: rgb_image} -> Estimate or None
"""
import time
from dataclasses import dataclass, field

from dronevision.l3_association.associator import DetectionAssociator
from dronevision.l5_estimation.smoothing import EMASmoother
from dronevision.l2_perception.detector import make_detector
from dronevision.l4_triangulation.geometry import Triangulator, default_site
from dronevision.io.schema import StateEstimate


@dataclass
class Estimate:
    """One pipeline output, with enough detail to diagnose a bad one."""

    position: tuple                  # smoothed, marker-corrected, world ENU (m)
    raw: tuple                       # before smoothing and correction
    cams_used: list                  # cameras surviving outlier rejection
    n_detections: int                # cameras that reported anything at all
    src_stamp: float = 0.0           # capture time of the frames used
    clock: str = "unknown"           # which clock src_stamp is on; see output.schema
    timings_ms: dict = field(default_factory=dict)

    @property
    def n_rejected(self):
        return self.n_detections - len(self.cams_used)

    def to_state(self, seq, quality=None):
        """Convert to the wire format that leaves the AI layer."""
        return StateEstimate(
            seq=seq,
            t=time.time(),
            src_t=self.src_stamp,
            pos_enu=tuple(self.position),
            cams=tuple(self.cams_used),
            n_fresh=self.n_detections,
            quality=(len(self.cams_used) / max(self.n_detections, 1)
                     if quality is None else quality),
            clock=self.clock,
            extra={"timings_ms": self.timings_ms} if self.timings_ms else {},
        )


class LocalizationPipeline:
    """Detect, associate, triangulate and smooth — one target, several cameras."""

    def __init__(self, cal=None, detector=None, associator=None, smoother=None,
                 occlude=None, cams=None, marker_correction=None, timing=False):
        self.cal = cal if cal is not None else default_site()
        self.detector = detector if detector is not None else make_detector()
        self.associator = associator or DetectionAssociator()
        self.smoother = smoother if smoother is not None else EMASmoother(0.5)
        self.tri = Triangulator(self.cal)
        self.cams = list(cams) if cams is not None else self.cal.cam_names
        #: Cameras to ignore. Used to reproduce occlusion without changing a world.
        self.occlude = set(occlude or ())
        # Whether the detected point sits above the airframe. Taken from the
        # detector unless overridden, because getting it wrong silently biases
        # every altitude by the offset rather than failing.
        self.marker_correction = (
            getattr(self.detector, "detects_marker", False)
            if marker_correction is None else bool(marker_correction))
        self.timing = timing
        self.seq = 0

    @property
    def active_cams(self):
        return [c for c in self.cams if c not in self.occlude]

    def detect(self, frames):
        """``{cam: rgb}`` -> ``{cam: (u, v)}``, one detection per camera."""
        dets = {}
        for cam in self.active_cams:
            img = frames.get(cam)
            if img is None:
                continue
            px = self.detector.detect(img, cam=cam)
            if px:
                dets[cam] = px
        return dets

    def locate(self, frames, src_stamp=0.0, clock="unknown"):
        """Run layers 2-5 on one set of frames. Returns an `Estimate` or None.

        None means too few cameras saw the target — a routine occurrence, not an
        error. The smoother is deliberately *not* reset on a miss: brief dropouts
        are common and the previous estimate remains the best available guess.
        """
        t = {}
        tick = time.perf_counter   # local alias; must not shadow the `clock` arg

        t0 = tick()
        dets = self.detect(frames)
        if self.timing:
            t["detect"] = (tick() - t0) * 1e3

        t0 = tick()
        views = self.associator.associate(dets).get(0, {})
        if self.timing:
            t["associate"] = (tick() - t0) * 1e3

        if len(views) < self.cal.min_views:
            return None

        t0 = tick()
        X, used = self.tri.triangulate(views)
        if self.timing:
            t["triangulate"] = (tick() - t0) * 1e3
        if X is None:
            return None

        t0 = tick()
        sm = self.smoother.update(X).copy()
        if self.marker_correction:
            sm[2] -= self.cal.marker_dz
        if self.timing:
            t["smooth"] = (tick() - t0) * 1e3

        self.seq += 1
        return Estimate(position=tuple(sm), raw=tuple(X), cams_used=used,
                        n_detections=len(dets), src_stamp=src_stamp,
                        clock=clock, timings_ms=t)

    def locate_from(self, source):
        """Convenience: pull the newest frame set from a `FrameSource` and locate.

        Uses the oldest stamp in the set, since that is the age of the estimate —
        it can be no fresher than its stalest input.

        Acquisition is TIMED, as an `acquire` stage. It is not free: a source may
        decode JPEG here, and on an Arm CPU that is a real part of the per-frame
        budget. Leaving it outside the measured region made the pipeline look
        identical whether frames arrived compressed or ready — which silently
        hid the cost the deployment actually pays.
        """
        t0 = time.perf_counter()
        frames = {c: source.latest(c) for c in self.active_cams}
        stamps = [m["stamp"] for c in self.active_cams
                  if (m := source.meta(c)) and m.get("stamp") is not None]
        acquire_ms = (time.perf_counter() - t0) * 1e3

        est = self.locate(frames, src_stamp=min(stamps) if stamps else 0.0,
                          clock=getattr(source, "clock", "unknown"))
        if est is not None and self.timing:
            est.timings_ms["acquire"] = acquire_ms
        return est

    def reset(self):
        self.smoother.reset()
