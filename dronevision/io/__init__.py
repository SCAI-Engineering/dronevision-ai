"""The AI block's boundaries — not one of the five layers.

    sources/       frames in: a published camera service, or a recorded corpus
    schema.py      the state estimate that leaves this block
    state_sink.py  sends it

Both boundaries are documented wire formats rather than shared code, so either
side can be reimplemented or replaced. That is what makes the drone, the cameras
and the control software external services rather than dependencies:

    frames in    ZeroMQ SUB, JPEG + JSON header   (sources/net.py)
    state out    UDP datagram, JSON               (schema.py)

Nothing here knows what a simulator is, and nothing here speaks to an autopilot.
"""

from dronevision.io.schema import SCHEMA_VERSION, StateEstimate
from dronevision.io.sources.base import FrameSource
from dronevision.io.state_sink import UdpStateSink

__all__ = ["FrameSource", "StateEstimate", "SCHEMA_VERSION", "UdpStateSink"]
