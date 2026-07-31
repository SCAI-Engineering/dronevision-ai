"""Cross-camera detection association.

With one drone in the scene this is trivial — a camera's single detection *is*
the target, so camera identity carries target identity — and `DetectionAssociator`
is a passthrough that wraps the detections as target 0.

The seam exists because the trivial case stops being trivial the moment a second
target appears, and the fix belongs here rather than smeared through
triangulation: gate candidate detections against each target's predicted position
reprojected into each camera, then assign within the gates, then carry stable
track identities across frames.
"""


class DetectionAssociator:
    """Map per-camera detections to target identities.

    Single-target passthrough. Multi-target assignment is not implemented.
    """

    def associate(self, dets_by_cam):
        """``{cam: (u, v)}`` -> ``{target_id: {cam: (u, v)}}``.

        Drops cameras reporting None, and returns an empty mapping rather than a
        target with no views, so callers can treat "nothing seen" uniformly.
        """
        views = {c: uv for c, uv in dets_by_cam.items() if uv is not None}
        return {0: views} if views else {}
