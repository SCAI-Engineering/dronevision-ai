"""Cross-camera frame alignment by timestamp — SCAFFOLD.

What runs is "newest frame from each camera", which is not time-aligned: a set can
mix frames captured milliseconds apart, and triangulating across them places the
target where it never was. With a slow target the error is small; it grows with
speed.

`FrameSync` is where real alignment would go — buffer recent frames per camera and
return a set consistent within a tolerance. It is not wired into any path.

Worth stating plainly: the scheduling approach this pipeline uses makes strict
alignment *less* important rather than more, because it stops pretending the
cameras were sampled simultaneously and instead carries each camera's estimate
forward in time individually (see `dronevision.l3_association.tracker`). Alignment
and per-camera tracking are alternative answers to the same problem, and only one
of them costs an inference.
"""


class FrameSync:
    """Collect per-camera frames and return a time-consistent set.

    Currently keeps only the newest frame per camera, i.e. present behaviour.
    """

    def __init__(self, cams, tol_ms=50.0):
        self.cams = list(cams)
        self.tol_ms = float(tol_ms)
        self._latest = {}                # cam -> (frame, stamp, seq)

    def add(self, cam, frame, stamp=None, seq=None):
        self._latest[cam] = (frame, stamp, seq)

    def latest_aligned(self):
        """``{cam: frame}``. Does not yet enforce the tolerance."""
        return {c: v[0] for c, v in self._latest.items() if v[0] is not None}

    def spread_ms(self):
        """Milliseconds between the oldest and newest buffered frame.

        Implemented even though alignment is not: this is the diagnostic that says
        whether the missing alignment actually matters for a given deployment.
        """
        stamps = [v[1] for v in self._latest.values() if v[1] is not None]
        if len(stamps) < 2:
            return 0.0
        return (max(stamps) - min(stamps)) * 1000.0
