#!/usr/bin/env python3
"""The AI process: camera frames in, 3D position out.

This is the whole deliverable running as one program. It consumes a camera service
and publishes a state estimate; it does not know what a simulator is, does not
speak to an autopilot, and does not care what consumes its output. That is why the
same command runs on a laptop and on a Raspberry Pi.

    python -m dronevision.service --frames tcp://192.168.1.50:5555 \
                                  --state 192.168.1.50:5601 --detector color

    python -m dronevision.service --replay data/corpus        # no network at all

Boundaries, in one place:

    in    camera service       ZeroMQ SUB   (see sync/sources/net.py)
    out   state estimate       UDP JSON     (see output/schema.py)

Both are documented formats, not shared code, so either side can be replaced.
"""
import argparse
import signal
import sys
import time

from dronevision.l5_estimation.smoothing import EMASmoother
from dronevision.io.state_sink import UdpStateSink
from dronevision.l2_perception.detector import make_detector
from dronevision.pipeline import LocalizationPipeline
from dronevision.l4_triangulation.calibration import load_site


def build_source(args, site):
    if args.replay:
        from dronevision.io.sources.replay import ReplaySource
        return ReplaySource(args.replay, loop=args.loop), True
    from dronevision.io.sources.net import NetSource
    src = NetSource(site.cam_names, endpoint=args.frames)
    print(f"[ai] waiting for cameras on {args.frames} ...", flush=True)
    if not src.wait_ready(timeout=args.wait):
        print(f"[ai] ERROR: no frames from {src.missing()}\n"
              f"     Is the camera service publishing? On the simulator host:\n"
              f"       tools/service/frame_publisher.py --bind tcp://0.0.0.0:5555",
              file=sys.stderr)
        raise SystemExit(1)
    return src, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="factory", help="site config name or path")
    ap.add_argument("--frames", default="tcp://127.0.0.1:5555",
                    help="camera service endpoint")
    ap.add_argument("--state", default="127.0.0.1:5601",
                    help="where to send state estimates; empty to not send")
    ap.add_argument("--detector", default=None, help="color | yolo | motion | hybrid")
    ap.add_argument("--rate", type=float, default=25.0, help="target Hz")
    ap.add_argument("--alpha", type=float, default=0.5, help="EMA smoothing 0..1")
    ap.add_argument("--occlude", default="", help="cameras to ignore, comma separated")
    ap.add_argument("--replay", default=None,
                    help="read a recorded corpus instead of the network")
    ap.add_argument("--loop", action="store_true", help="loop a replayed corpus")
    ap.add_argument("--wait", type=float, default=30.0, help="camera wait timeout (s)")
    ap.add_argument("--timing", action="store_true", help="report per-stage timings")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    site = load_site(a.site)
    src, replaying = build_source(a, site)
    sink = UdpStateSink(a.state) if a.state else None

    pipe = LocalizationPipeline(
        cal=site, detector=make_detector(a.detector), smoother=EMASmoother(a.alpha),
        occlude=[c for c in a.occlude.split(",") if c], timing=a.timing or not a.quiet)

    print(f"[ai] site={site.name} detector={pipe.detector.name} "
          f"cameras={pipe.active_cams}", flush=True)
    print(f"[ai] state -> {a.state or '(not sent)'}"
          f"{'  [replay]' if replaying else ''}", flush=True)

    running = {"go": True}
    signal.signal(signal.SIGINT, lambda *_: running.update(go=False))

    period = 1.0 / a.rate
    n = miss = 0
    t_start = time.time()
    ms_total = 0.0
    last_warn = 0.0

    while running["go"]:
        t0 = time.time()
        if replaying and not src.step():
            break

        est = pipe.locate_from(src)
        if est is None:
            miss += 1
            # Distinguish "cannot see the target" from "no longer receiving
            # frames". Both produce no estimate, but only the second means the
            # input has died — and staying silent about that is how a frozen
            # position goes unnoticed for minutes.
            stale = src.stale() if hasattr(src, "stale") else []
            if stale and time.time() - last_warn > 2.0:
                print(f"[ai] NO FRAMES from {stale} — camera service down? "
                      f"emitting nothing", file=sys.stderr, flush=True)
                last_warn = time.time()
        else:
            n += 1
            ms_total += sum(est.timings_ms.values())
            if sink is not None:
                sink.send(est.to_state(seq=n))
            if not a.quiet and n % 25 == 0:
                e, nn, u = est.position
                extra = ""
                if est.timings_ms:
                    extra = "  %.2f ms" % sum(est.timings_ms.values())
                print("[%6d] ENU=(%+.2f,%+.2f,%+.2f) cams=%d/%d%s"
                      % (n, e, nn, u, len(est.cams_used), len(pipe.active_cams),
                         extra), flush=True)
        if not replaying:
            time.sleep(max(0.0, period - (time.time() - t0)))

    elapsed = time.time() - t_start
    print(f"\n[ai] {n} estimates, {miss} misses, {elapsed:.1f} s "
          f"({n / elapsed:.1f} Hz achieved)", flush=True)
    if n:
        print(f"[ai] pipeline mean {ms_total / n:.2f} ms "
              f"-> {1000 * n / ms_total:.0f} Hz compute ceiling", flush=True)
    if sink is not None:
        print(f"[ai] sent {sink.sent} datagrams ({sink.failed} failed)", flush=True)
        sink.close()
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
