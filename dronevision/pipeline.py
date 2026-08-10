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
    src_stamp: float = 0.0           # OLDEST capture time in the set: a staleness
                                     # bound, deliberately pessimistic
    newest_stamp: float = 0.0        # NEWEST capture time in the set
    # MEAN capture time -- the instant this estimate actually describes, and the
    # one to use when lining it up against any other time series.
    #
    # The three are not interchangeable. `src_stamp` answers "how old could this
    # be?" (what a consumer deciding whether to act on it needs) and
    # `newest_stamp` bounds the other end; neither is when the scene happened.
    # Triangulation weights every camera roughly equally, so the 3D point it
    # returns corresponds to the average of their capture instants, not to
    # either extreme. With real cross-camera skew the extremes sit tens of
    # milliseconds apart -- 80 ms on the 2 m/s corpus, 160 mm of target movement
    # -- so indexing by an endpoint charges half that gap to the estimate as if
    # it were error. Measured: on three corpora the error-minimising instant
    # landed within 5 ms of this mean every time, while both endpoints were
    # 10-55 mm worse.
    mean_stamp: float = 0.0
    clock: str = "unknown"           # which clock the stamps are on; see output.schema
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
                 occlude=None, cams=None, marker_correction=None, timing=False,
                 sync=None):
        self.cal = cal if cal is not None else default_site()
        self.detector = detector if detector is not None else make_detector()
        self.associator = associator or DetectionAssociator()
        self.smoother = smoother if smoother is not None else EMASmoother(0.5)
        self.tri = Triangulator(self.cal)
        #: Optional `FrameSync` (layer 1). None keeps today's behaviour: the
        #: newest frame per camera, whatever its age relative to the others.
        #: Given one, `locate_from` routes acquisition through it and drops
        #: cameras that fell out of alignment instead of triangulating them
        #: as if they were simultaneous. See `l1_image_sync.aligner`.
        self.sync = sync
        self.cams = list(cams) if cams is not None else self.cal.cam_names
        #: Cameras to ignore. Used to reproduce occlusion without changing a world.
        self.occlude = set(occlude or ())
        # How far above the vehicle origin the detected point sits. Taken from the
        # detector's own declaration and looked up in the site config, because the
        # detectors do not look at the same thing: a colour marker sits 0.4333 m up on
        # this site, a bounding-box centroid 0.0950 m. Getting it wrong never raises —
        # it shifts every altitude by a constant and leaves horizontal accuracy intact.
        if marker_correction is None:
            key = getattr(self.detector, "target_offset_key", None)
        elif marker_correction:
            key = "marker_dz"
        else:
            key = None
        self.target_offset_key = key
        self.target_dz = self.cal.target_offset(key)
        self.marker_correction = bool(key)      # older callers read this
        self.timing = timing
        self.seq = 0

    @property
    def active_cams(self):
        return [c for c in self.cams if c not in self.occlude]

    def detect(self, frames):
        """``{cam: rgb}`` -> ``{cam: (u, v)}``, one detection per camera.

        Delegates wholesale when the detector can handle the whole set itself — that is
        how camera-parallel perception plugs in without the pipeline knowing about
        threads or cores.
        """
        active = {c: frames.get(c) for c in self.active_cams}
        detect_all = getattr(self.detector, "detect_all", None)
        if detect_all is not None:
            return detect_all({c: img for c, img in active.items() if img is not None})

        dets = {}
        for cam, img in active.items():
            if img is None:
                continue
            px = self.detector.detect(img, cam=cam)
            if px:
                dets[cam] = px
        return dets

    def locate(self, frames, src_stamp=0.0, clock="unknown", newest_stamp=None,
               mean_stamp=None):
        """Run layers 2-5 on one set of frames. Returns an `Estimate` or None.

        None means too few cameras saw the target — a routine occurrence, not an
        error. The smoother is deliberately *not* reset on a miss: brief dropouts
        are common and the previous estimate remains the best available guess.

        `newest_stamp` and `mean_stamp` both default to `src_stamp`, which is
        right for a caller that has only one time to give: with no skew
        information all three coincide.
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
        sm[2] -= self.target_dz
        if self.timing:
            t["smooth"] = (tick() - t0) * 1e3

        self.seq += 1
        return Estimate(position=tuple(sm), raw=tuple(X), cams_used=used,
                        n_detections=len(dets), src_stamp=src_stamp,
                        newest_stamp=src_stamp if newest_stamp is None
                        else newest_stamp,
                        mean_stamp=src_stamp if mean_stamp is None
                        else mean_stamp,
                        clock=clock, timings_ms=t)

    def locate_from(self, source):
        """Convenience: pull the newest frame set from a `FrameSource` and locate.

        Records both ends of the set's capture window. The oldest stamp is the
        age of the estimate — it can be no fresher than its stalest input — and
        the newest is when the scene it describes actually happened. Anything
        lining the estimate up against another time series wants the second; a
        consumer deciding whether the position is too old to act on wants the
        first.

        Acquisition is TIMED, as an `acquire` stage. It is not free: a source may
        decode JPEG here, and on an Arm CPU that is a real part of the per-frame
        budget. Leaving it outside the measured region made the pipeline look
        identical whether frames arrived compressed or ready — which silently
        hid the cost the deployment actually pays.

        With `self.sync` set, every active camera's newest frame is fed to it
        before asking for the aligned set, so a camera that fell out of alignment
        is dropped here rather than reaching triangulation.
        """
        t0 = time.perf_counter()
        if self.sync is not None:
            for c in self.active_cams:
                frame = source.latest(c)
                if frame is not None:
                    meta = source.meta(c) or {}
                    self.sync.add(c, frame, stamp=meta.get("stamp"),
                                  seq=meta.get("seq"))
            frames = self.sync.latest_aligned()
            stamps = [self.sync.stamp(c) for c in frames]
        else:
            frames = {c: source.latest(c) for c in self.active_cams}
            stamps = [m["stamp"] for c in self.active_cams
                      if (m := source.meta(c)) and m.get("stamp") is not None]
        acquire_ms = (time.perf_counter() - t0) * 1e3

        est = self.locate(frames, src_stamp=min(stamps) if stamps else 0.0,
                          newest_stamp=max(stamps) if stamps else 0.0,
                          mean_stamp=(sum(stamps) / len(stamps)) if stamps else 0.0,
                          clock=getattr(source, "clock", "unknown"))
        if est is not None and self.timing:
            est.timings_ms["acquire"] = acquire_ms
        return est

    def reset(self):
        self.smoother.reset()
