"""The state estimate leaving the AI layer — the contract with the consumer.

Small, flat, and versioned, so it survives a UDP datagram and a schema change.

    {"v": 1, "seq": 4821, "t": 1748...567, "src_t": 1234.501,
     "pos_enu": [1.20, -0.42, 2.44], "cams": ["cam_ne", "cam_sw"],
     "n_fresh": 1, "quality": 0.87}

WHY BOTH TIMESTAMPS: `src_t` is the capture time of the images this estimate came
from, on the camera's clock; `t` is when we emitted it, on ours. A controller
fusing a position it believes is current, when it is actually 80 ms old, will
oscillate — and the fix is a delay compensation parameter that someone has to
supply a number for. Shipping both timestamps means that number is measured
rather than guessed. It costs 8 bytes.

COORDINATES: `pos_enu` is metres in the room frame, East-North-Up, origin at the
room centre — the same frame the camera calibration is expressed in. Conversion
to whatever convention the consumer wants (e.g. North-East-Down) belongs on the
consumer's side, so this package stays free of flight-stack assumptions.
"""
import json
from dataclasses import dataclass, field

#: Bump when the wire format changes incompatibly. Consumers should reject
#: unknown majors rather than silently misread a field.
SCHEMA_VERSION = 1


@dataclass
class StateEstimate:
    """One 3D position estimate, with the provenance needed to trust it."""

    seq: int                      # monotonic, per run; gaps mean dropped estimates
    t: float                      # emit time, local wall clock (unix seconds)
    src_t: float                  # capture time of the source frames
    pos_enu: tuple                # (east, north, up) in metres
    cams: tuple = ()              # cameras that contributed after outlier rejection
    n_fresh: int = 0              # how many of those were fresh detections, not coasted
    quality: float = 0.0          # 0..1 confidence; see `dronevision.l5_estimation`
    # Which clock `src_t` is on. "unix" means it shares an epoch with `t`, so the
    # difference is a real latency; "sim" means it is simulation time and the
    # difference is meaningless. Without this field a consumer subtracts two
    # unrelated epochs and prints a confident nonsense number — which is exactly
    # what happened before it existed.
    clock: str = "unknown"
    extra: dict = field(default_factory=dict)   # diagnostics; consumers may ignore

    @property
    def latency(self):
        """Seconds between image capture and emission, or None if incomparable.

        This is the number a controller needs to compensate for delay, so
        returning None beats returning a wrong one.
        """
        if self.clock != "unix":
            return None
        return self.t - self.src_t

    def to_dict(self):
        d = {
            "v": SCHEMA_VERSION,
            "seq": self.seq,
            "t": round(self.t, 6),
            "src_t": round(self.src_t, 6),
            "pos_enu": [round(float(v), 4) for v in self.pos_enu],
            "cams": list(self.cams),
            "n_fresh": self.n_fresh,
            "quality": round(float(self.quality), 3),
            "clk": self.clock,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    def to_json(self):
        """Compact JSON, sized to fit one datagram."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d):
        v = d.get("v")
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported state schema v{v} (this build speaks v{SCHEMA_VERSION})")
        return cls(
            seq=int(d["seq"]),
            t=float(d["t"]),
            src_t=float(d["src_t"]),
            pos_enu=tuple(float(x) for x in d["pos_enu"]),
            cams=tuple(d.get("cams", ())),
            n_fresh=int(d.get("n_fresh", 0)),
            quality=float(d.get("quality", 0.0)),
            clock=str(d.get("clk", "unknown")),
            extra=dict(d.get("extra", {})),
        )

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))
