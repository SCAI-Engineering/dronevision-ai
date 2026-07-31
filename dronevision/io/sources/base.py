"""The `FrameSource` contract — the AI layer's only camera input.

Deliberately the smallest interface that supports every deployment mode, and
deliberately identical in shape to the simulator's original camera reader, so
adopting it requires no changes downstream.

Frames are RGB uint8, HxWx3 — matching what the simulator publishes. Sources
convert to that if their transport differs (e.g. JPEG on the wire).

`meta()` carries the frame's own timestamp, which matters more than it looks:
the consumer of our state estimate has to know how stale the underlying image
was, or a closed control loop will oscillate. See `dronevision.io.schema`.
"""
from abc import ABC, abstractmethod


class FrameSource(ABC):
    """Latest-frame-per-camera access to a set of fixed cameras.

    Implementations keep the newest frame per camera and overwrite it as new ones
    arrive; a slow consumer sees fresh frames, never a backlog. That is the right
    trade for localization, where a stale frame has negative value.
    """

    #: Camera names, in a stable order. Ordering is meaningful — the scheduler in
    #: layer 1 rotates through this list.
    cams: list

    #: Which clock the stamps from `meta()` are on: "unix" if it shares an epoch
    #: with local wall clock, "sim" for simulation time, "unknown" otherwise. The
    #: consumer of our estimate needs this to know whether a measured latency is
    #: real, and a source that lies here causes a controller to compensate for a
    #: delay that does not exist.
    clock: str = "unknown"

    @abstractmethod
    def latest(self, cam):
        """Newest frame for `cam` as RGB uint8 HxWx3, or None if none has arrived."""

    @abstractmethod
    def meta(self, cam):
        """Metadata for the newest frame of `cam`, or None.

        Returns ``{"stamp": float, "seq": int | None, "frame_id": str | None}``
        where ``stamp`` is the capture time in seconds on the *source's* clock
        (simulation time for simulated cameras) — not local wall clock.
        """

    def all(self):
        """``{cam: frame_or_None}`` for every camera."""
        return {c: self.latest(c) for c in self.cams}

    def close(self):
        """Release transport resources. Safe to call more than once."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
