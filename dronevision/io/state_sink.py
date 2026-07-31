"""Send the state estimate out of the AI, over UDP.

The AI's output boundary. It emits a position and stops; what happens next is the
control software's business, and this module knows nothing about flight stacks,
MAVLink, or coordinate conventions other than its own.

UDP and not TCP, deliberately: each estimate is small, self-contained, and
worthless once superseded. A retransmitted position from 200 ms ago is not useful
to a controller, so head-of-line blocking would be strictly harmful.
"""
import socket

from dronevision.io.schema import StateEstimate


class UdpStateSink:
    """Fire-and-forget sink for `StateEstimate`.

    Never raises on a send failure. A missing consumer must not take the AI down —
    localization is still worth doing (and worth logging) with nobody listening.
    """

    def __init__(self, endpoint="127.0.0.1:5601"):
        host, _, port = endpoint.rpartition(":")
        self.addr = (host or "127.0.0.1", int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.failed = 0

    def send(self, est):
        """Send a `StateEstimate` (or a pre-built dict). True if it went out."""
        payload = (est.to_json() if isinstance(est, StateEstimate)
                   else StateEstimate.from_dict(est).to_json())
        try:
            self.sock.sendto(payload.encode("utf-8"), self.addr)
            self.sent += 1
            return True
        except OSError:
            self.failed += 1
            return False

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
