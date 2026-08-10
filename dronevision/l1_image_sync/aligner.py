"""Cross-camera frame alignment by timestamp.

What ran before this: "newest frame from each camera", which is not time-aligned —
a set can mix frames captured milliseconds apart, and triangulating across them
places the target where it never was. With a slow target the error is small; it
grows with speed. Measured on a corpus recorded at 2 m/s with hard cornering: camera
capture stamps drift up to 84 ms apart within one "simultaneous" set (cameras share
no clock even in simulation — gz's own per-sensor render scheduling staggers them
under load, and the drift is systematic rather than random: one camera lagged more
than 50 ms behind the rest in 91% of frames), which alone accounts for roughly
150 mm of the ~240 mm 3D error seen at that speed.

`FrameSync` buffers the newest frame per camera and returns the *largest set of
cameras that are mutually within `tol_ms` of each other* — not simply whichever
cameras are close to the single newest frame. That distinction matters in practice:
on the corpus above, anchoring to the newest frame left at least two aligned
cameras only 65% of the time, because the newest frame was often an isolated
straggler-of-one while the other three sat tightly together, just further back.
Searching for the best-aligned *cluster* recovered a usable set 100% of the time
on the same data, from the same buffered frames — the earlier version was
discarding good data along with the bad.

Worth stating plainly: the scheduling approach elsewhere in this pipeline makes
strict alignment *less* important rather than more, because it stops pretending
the cameras were sampled simultaneously and instead carries each camera's estimate
forward in time individually (see `dronevision.l3_association.tracker`). Alignment
and per-camera tracking are alternative answers to the same problem, and only one
of them costs an inference.
"""
from itertools import combinations


class FrameSync:
    """Collect per-camera frames and return a time-consistent set.

    `latest_aligned()` returns the largest subset of buffered cameras whose
    pairwise capture-time spread is within `tol_ms` — not whichever cameras
    happen to be close to the single newest one. A camera left out of that
    subset is excluded rather than triangulated as if it were simultaneous with
    the rest.
    """

    def __init__(self, cams, tol_ms=50.0, max_age_ms=200.0):
        self.cams = list(cams)
        self.tol_ms = float(tol_ms)
        #: How far behind the newest buffered frame a camera may be and still be
        #: considered at all. Buffered frames are never evicted, so a camera that
        #: stops publishing keeps its last frame here indefinitely — and since
        #: groups are ranked by spread first, a pair that went stale *together*
        #: stays perfectly consistent with itself and would keep beating a fresher
        #: but slightly looser pair, forever. Alignment alone cannot catch that:
        #: being mutually consistent and being current are different properties.
        self.max_age_ms = float(max_age_ms)
        self._latest = {}                # cam -> (frame, stamp, seq)

    def add(self, cam, frame, stamp=None, seq=None):
        self._latest[cam] = (frame, stamp, seq)

    def latest_aligned(self):
        """``{cam: frame}``, the largest mutually time-consistent subset.

        Cameras with no timestamp are dropped first: alignment cannot be judged
        without one, and an unverifiable frame is worse than none here. So are
        cameras more than `max_age_ms` behind the newest buffered frame, before
        any grouping — otherwise a set that went stale together would rank above
        a fresher one, being tightly consistent with itself. Among what remains,
        every group size from "all buffered cameras" down to pairs is checked
        (four cameras: at most 11 groups — the same exhaustive-rather-than-sampled
        trade `l4_triangulation.geometry._consensus` makes, and cheap for the same
        reason). The largest size that satisfies the tolerance wins; among groups
        of that size the tighter spread wins, then the more recent one. If no pair
        aligns, the single newest frame is returned alone — the same fallback a
        lone buffered camera gets.
        """
        stamped = {c: v for c, v in self._latest.items()
                  if v[0] is not None and v[1] is not None}
        if len(stamped) >= 2:
            newest_all = max(v[1] for v in stamped.values())
            horizon = self.max_age_ms / 1000.0
            stamped = {c: v for c, v in stamped.items()
                       if newest_all - v[1] <= horizon}
        if len(stamped) < 2:
            return {c: v[0] for c, v in stamped.items()}

        tol = self.tol_ms / 1000.0
        names = list(stamped)
        for k in range(len(names), 1, -1):
            groups = []
            for combo in combinations(names, k):
                vals = [stamped[c][1] for c in combo]
                spread = max(vals) - min(vals)
                if spread <= tol:
                    groups.append((combo, spread, max(vals)))
            if groups:
                # Tightest spread wins; among equally tight groups, the more
                # recent one -- a consistent-but-stale set is still stale.
                combo, _, _ = min(groups, key=lambda g: (g[1], -g[2]))
                return {c: stamped[c][0] for c in combo}

        newest = max(stamped, key=lambda c: stamped[c][1])
        return {newest: stamped[newest][0]}

    def stamp(self, cam):
        """Capture timestamp of the buffered frame for `cam`, or None."""
        v = self._latest.get(cam)
        return v[1] if v else None

    def spread_ms(self):
        """Milliseconds between the oldest and newest buffered frame.

        Deliberately includes cameras `latest_aligned()` would now exclude: this
        is the diagnostic for how much alignment is actually discarding, not a
        restatement of its output.
        """
        stamps = [v[1] for v in self._latest.values() if v[1] is not None]
        if len(stamps) < 2:
            return 0.0
        return (max(stamps) - min(stamps)) * 1000.0
