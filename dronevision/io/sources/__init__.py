"""Frame sources — one per deployment mode, all satisfying `FrameSource`.

    NetSource     subscribes to a published camera service (ZeroMQ). The live
                  input, on a laptop or a Raspberry Pi alike.
    ReplaySource  reads a recorded corpus from disk. Deterministic, faster than
                  real time, and needs no camera service at all.

There is no simulator-specific source, on purpose. The simulator publishes a
camera service like any other camera would, and this package consumes it — so
nothing here imports Gazebo, and the code that runs against the simulator is
byte-for-byte the code that runs on hardware.

Because every source presents the same three methods, swapping one for another
changes no code downstream: the pipeline cannot tell which it is talking to.
"""

from dronevision.io.sources.base import FrameSource

__all__ = ["FrameSource"]
