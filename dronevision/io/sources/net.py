"""Camera frames from the network — the AI's only live input.

Consumes a camera service (the simulator publishes one; real hardware would
publish the same thing) over ZeroMQ SUB. The AI never touches a simulator API, so
this same class is what runs on a Raspberry Pi.

    with NetSource(["cam_ne", "cam_nw", "cam_sw", "cam_se"]) as src:
        src.wait_ready()
        est = pipeline.locate_from(src)

LATEST WINS, ALWAYS. `zmq.CONFLATE` keeps only the newest message per topic, so a
consumer slower than the publisher sees fresh frames rather than working through a
backlog. For localization a stale frame is worse than no frame: it reports the
target where it used to be, confidently.

Ground truth, when the service publishes it, arrives on its own topic and is
exposed separately — it is measurement data, never an input to the estimate.
"""
import json
import threading
import time

import numpy as np

from dronevision.io.sources.base import FrameSource

TRUTH_TOPIC = "truth"


class NetSource(FrameSource):
    """Subscribe to a published camera service and hold the newest frame each."""

    def __init__(self, cams, endpoint="tcp://127.0.0.1:5555", truth=True,
                 decode=True, max_age=0.5):
        try:
            import zmq
        except ImportError as e:
            raise SystemExit(
                f"pyzmq required for NetSource ({e}); install with: "
                f"pip install 'dronevision[net]'")

        self.cams = list(cams)
        self.endpoint = endpoint
        self.decode = decode
        #: Frames older than this (seconds since arrival) are reported as absent.
        #: Generous relative to a 25 Hz service, so ordinary jitter never trips it,
        #: while a dead publisher is caught within half a second. None disables the
        #: check — only sensible when replaying, never on a live loop.
        self.max_age = max_age
        self._frames = {}
        self._meta = {}
        self._arrived = {}                  # cam -> monotonic arrival time
        self._truth = None
        self._counts = {c: 0 for c in self.cams}
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        # NOT zmq.CONFLATE. It is the obvious choice for "keep only the newest"
        # and it silently discards every multipart message, which is all of them
        # here (topic + header + payload). Latest-wins is achieved instead by the
        # reader thread below, which drains continuously and overwrites — same
        # semantics, and it works.
        #
        # A shallow queue so a stalled reader drops frames rather than growing:
        # for localization a stale frame is worse than no frame.
        self._sock.setsockopt(zmq.RCVHWM, 8)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        for cam in self.cams:
            self._sock.setsockopt(zmq.SUBSCRIBE, cam.encode())
        if truth:
            self._sock.setsockopt(zmq.SUBSCRIBE, TRUTH_TOPIC.encode())
        self._sock.connect(endpoint)

        # A background reader, so the pipeline's own timing never includes waiting
        # on the wire — the two would otherwise be impossible to separate.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- receive ------------------------------------------------------------

    def _run(self):
        import zmq
        while not self._stop.is_set():
            try:
                parts = self._sock.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if len(parts) < 2:
                continue
            topic = parts[0].decode("utf-8", "replace")
            try:
                header = json.loads(parts[1].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

            if topic == TRUTH_TOPIC:
                with self._lock:
                    self._truth = header
                continue

            if topic not in self._counts or len(parts) < 3:
                continue
            payload = parts[2]
            img = self._decode(payload) if self.decode else payload
            with self._lock:
                self._frames[topic] = img
                self._arrived[topic] = time.monotonic()
                self._meta[topic] = {"stamp": header.get("stamp"),
                                     "seq": header.get("seq"),
                                     "frame_id": topic}
                # Taken from the service rather than assumed: only the publisher
                # knows whether its stamps share an epoch with our wall clock.
                self.clock = header.get("clock", "unknown")
                self._counts[topic] += 1

    @staticmethod
    def _decode(payload):
        import cv2
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -- FrameSource --------------------------------------------------------

    def latest(self, cam):
        """Newest frame for `cam`, or None if it has gone stale.

        STALENESS IS NOT OPTIONAL HERE. A publisher that dies leaves the last frame
        sitting in memory, and without this check the pipeline keeps triangulating
        it — reporting a confident, frozen position for as long as the process runs.
        Downstream that is worse than reporting nothing: a controller told the
        vehicle is stationary at a stale point will fly away from it. Observed for
        real when the camera service crashed and the estimate never changed.
        """
        with self._lock:
            if self._is_stale(cam):
                return None
            return self._frames.get(cam)

    def meta(self, cam):
        with self._lock:
            if self._is_stale(cam):
                return None
            m = self._meta.get(cam)
            return dict(m) if m else None

    def _is_stale(self, cam):
        """Caller holds the lock. Measured against ARRIVAL time, not the frame's
        own stamp: the stamp may be on a simulation clock, and the question here is
        whether the transport is still delivering, not what time the sensor thinks
        it is."""
        if self.max_age is None:
            return False
        t = self._arrived.get(cam)
        return t is None or (time.monotonic() - t) > self.max_age

    def stale(self):
        """Cameras whose last frame is older than `max_age`."""
        with self._lock:
            return [c for c in self.cams if self._is_stale(c)]

    def age_s(self, cam):
        """Seconds since a frame last arrived for `cam`, or None if never."""
        with self._lock:
            t = self._arrived.get(cam)
        return None if t is None else time.monotonic() - t

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._sock.close(linger=0)
            self._ctx.term()
        except Exception:
            pass

    # -- status -------------------------------------------------------------

    def wait_ready(self, timeout=30.0, poll=0.05):
        """Block until every camera has delivered at least one frame."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.missing():
                return True
            time.sleep(poll)
        return False

    def missing(self):
        with self._lock:
            return [c for c in self.cams if c not in self._frames]

    @property
    def counts(self):
        """Frames received per camera. A camera stuck at 0 is a naming mismatch."""
        with self._lock:
            return dict(self._counts)

    @property
    def truth(self):
        """Latest published vehicle pose as an ndarray, or None.

        Measurement data only. Nothing in the pipeline may read this — an
        estimator that can see the answer is not being measured.
        """
        with self._lock:
            t = self._truth
        if not t or "pos" not in t:
            return None
        return np.array(t["pos"], float)

    @property
    def truth_stamp(self):
        with self._lock:
            return None if not self._truth else self._truth.get("stamp")

    def skew_ms(self):
        """Spread between the newest and oldest frame currently held.

        The cameras do not publish in lockstep; this reports by how much, which is
        what layer 1 exists to deal with.
        """
        with self._lock:
            stamps = [m["stamp"] for m in self._meta.values()
                      if m.get("stamp") is not None]
        if len(stamps) < 2:
            return 0.0
        return (max(stamps) - min(stamps)) * 1000.0
