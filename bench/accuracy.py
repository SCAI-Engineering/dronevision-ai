#!/usr/bin/env python3
"""How accurate is the AI, and how long does it take?

Runs against a recorded corpus by default — no camera service, no vehicle, no
simulator — so the numbers are reproducible anywhere, including on the Arm device
this is being optimized for. `--frames` measures live instead.

    python -m bench.accuracy                              # from data/corpus
    python -m bench.accuracy --detector yolo
    python -m bench.accuracy --frames tcp://127.0.0.1:5555 -n 200
    python -m bench.accuracy --occlude cam_ne,cam_sw
    python -m bench.accuracy --marker-offset               # calibrate the offset

READING THE VERTICAL ERROR. It is reported signed. A large mean with a small
spread means the marker offset is wrong, not that the estimate is noisy — two very
different problems that look identical in a 3D error figure. `--marker-offset`
measures the offset directly and suggests the config value.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dronevision.l5_estimation.smoothing import EMASmoother
from dronevision.l2_perception.detector import make_detector
from dronevision.pipeline import LocalizationPipeline
from dronevision.l4_triangulation.calibration import load_site
from dronevision.l4_triangulation.geometry import triangulate


def stats_mm(values):
    a = np.asarray(list(values), float) * 1000
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": a.mean(), "median": np.median(a),
            "p95": np.percentile(a, 95), "max": a.max()}


def open_source(a, site):
    if a.frames:
        from dronevision.io.sources.net import NetSource
        src = NetSource(site.cam_names, endpoint=a.frames)
        print(f"waiting for cameras on {a.frames} ...")
        if not src.wait_ready(timeout=30):
            raise SystemExit(f"no frames from {src.missing()}; is the camera "
                             f"service publishing?")
        return src, True
    from dronevision.io.sources.replay import ReplaySource
    return ReplaySource(a.corpus, mode=a.mode), False


def samples(src, live, limit, interval):
    """Yield (source, truth, speed) for each measurement point."""
    if not live:
        for s in src:
            yield s, s.truth, s.truth_speed
        return
    prev, prev_t = None, None
    for _ in range(limit):
        t = src.truth
        speed = None
        now = time.time()
        if t is not None and prev is not None and now > prev_t:
            speed = float(np.linalg.norm(t - prev) / (now - prev_t))
        if t is not None:
            prev, prev_t = t.copy(), now
        yield src, t, speed
        time.sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--frames", default=None, help="measure live from this endpoint")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--detector", default="color")
    ap.add_argument("--runtime", default=None,
                    help="yolo only: ultralytics | onnx | executorch")
    ap.add_argument("--threads", type=int, default=None,
                    help="pin inference threads; required for comparable timings")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="network input size; must match across runtimes or the "
                         "comparison measures different amounts of work")
    ap.add_argument("--parallel", type=int, nargs="?", const=-1, default=None,
                    metavar="N", help="detect on N cameras concurrently")
    ap.add_argument("--mode", default="decoded", help="decoded | encoded | stream")
    ap.add_argument("-n", "--samples", type=int, default=200, help="live only")
    ap.add_argument("--interval", type=float, default=0.05, help="live only")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="EMA smoothing; 0 so accuracy is measured raw")
    ap.add_argument("--occlude", default="")
    ap.add_argument("--marker-offset", action="store_true",
                    help="measure the detected point's offset from truth")
    ap.add_argument("--max-speed", type=float, default=0.15,
                    help="--marker-offset: discard faster samples (motion blur "
                         "drags the centroid and biases the offset)")
    a = ap.parse_args(argv)

    site = load_site(a.site)
    src, live = open_source(a, site)
    det_kw = {}
    if a.detector in ("yolo", "hybrid"):
        det_kw = {"runtime": a.runtime, "threads": a.threads, "imgsz": a.imgsz}
    if a.parallel is not None:
        from dronevision.l2_perception.parallel import make_parallel
        detector = make_parallel(
            site.cam_names, detector=a.detector,
            workers=None if a.parallel < 0 else a.parallel,
            **{k: v for k, v in det_kw.items() if k != "threads"})
    else:
        detector = make_detector(a.detector, **det_kw)
    occlude = [c for c in a.occlude.split(",") if c]
    pipe = LocalizationPipeline(cal=site, detector=detector,
                                smoother=EMASmoother(a.alpha),
                                occlude=occlude, timing=True)

    print(f"site={site.name} detector={detector.name} "
          f"marker_correction={pipe.marker_correction}")
    print(f"source={'live ' + a.frames if live else a.corpus} "
          f"cameras={pipe.active_cams}" + (f" occluded={occlude}" if occlude else ""))
    print()

    err3, errz, errxy, ncams, nrej, alts, speeds = [], [], [], [], [], [], []
    offsets, off_alts = [], []
    stage = {}
    miss = 0
    t_start = time.time()

    for s, truth, speed in samples(src, live, a.samples, a.interval):
        est = pipe.locate_from(s)
        if est is None or truth is None:
            miss += 1
            continue
        d = np.array(est.position) - truth
        err3.append(float(np.linalg.norm(d)))
        errxy.append(float(np.linalg.norm(d[:2])))
        errz.append(float(d[2]))
        ncams.append(len(est.cams_used))
        nrej.append(est.n_rejected)
        alts.append(float(truth[2]))
        if speed is not None:
            speeds.append(speed)
        for k, v in est.timings_ms.items():
            stage.setdefault(k, []).append(v)

        if a.marker_offset and (speed is None or speed <= a.max_speed):
            dets = {}
            for cam in pipe.active_cams:
                img = s.latest(cam)
                if img is None:
                    continue
                uv = detector.detect(img, cam=cam)
                if uv:
                    dets[cam] = uv
            if len(dets) >= site.min_views:
                X, _ = triangulate(dets, cal=site)   # uncorrected detected point
                offsets.append(X - truth)
                off_alts.append(float(truth[2]))

    elapsed = time.time() - t_start
    if not err3:
        raise SystemExit(f"no estimates ({miss} misses) — is the vehicle visible "
                         f"to at least {site.min_views} cameras?")

    s3, sxy = stats_mm(err3), stats_mm(errxy)
    az = np.array(errz) * 1000
    print("samples %d   misses %d   altitude %.2f-%.2f m   wall %.2f s"
          % (s3["n"], miss, min(alts), max(alts), elapsed))
    if speeds:
        print("speed    mean %.3f m/s   max %.3f m/s" % (np.mean(speeds), max(speeds)))
    print()
    print("ERROR vs ground truth (mm)")
    print("  3D         mean %7.2f  median %7.2f  p95 %7.2f  max %7.2f"
          % (s3["mean"], s3["median"], s3["p95"], s3["max"]))
    print("  horizontal mean %7.2f  median %7.2f  p95 %7.2f  max %7.2f"
          % (sxy["mean"], sxy["median"], sxy["p95"], sxy["max"]))
    print("  vertical   mean %+7.2f  sd %6.2f" % (az.mean(), az.std()))
    print("             (large |mean| with small sd -> marker_dz is wrong, not noisy)")
    print()
    print("CAMERAS  used mean %.2f / %d   rejected %d"
          % (np.mean(ncams), len(pipe.active_cams), sum(nrej)))
    print()
    print("TIMING (ms per estimate)")
    for k in ("acquire", "detect", "associate", "triangulate", "smooth"):
        if k in stage:
            v = np.array(stage[k])
            print("  %-12s mean %8.3f  p95 %8.3f  max %8.3f"
                  % (k, v.mean(), np.percentile(v, 95), v.max()))
    if stage:
        tot = np.array([sum(v) for v in zip(*stage.values())])
        print("  %-12s mean %8.3f   -> %.0f Hz compute ceiling"
              % ("TOTAL", tot.mean(), 1000.0 / tot.mean()))

    if speeds and len(err3) > 20:
        print()
        print("ERROR vs SPEED (mm)")
        r = np.array([(sp, e * 1000) for sp, e in
                      zip(speeds, err3[-len(speeds):])])
        for lo, hi, label in [(0, .05, "still  <0.05"), (.05, .2, "slow  .05-.2"),
                              (.2, .6, "moving .2-.6"), (.6, 99, "fast    >0.6")]:
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.sum() >= 3:
                print("  %-14s n=%-5d mean %7.2f  p95 %7.2f"
                      % (label, m.sum(), r[m, 1].mean(), np.percentile(r[m, 1], 95)))

    if a.marker_offset:
        print()
        if len(offsets) < 5:
            print("MARKER OFFSET: only %d still samples; hold the vehicle steady"
                  % len(offsets))
        else:
            o = np.array(offsets)
            span = max(off_alts) - min(off_alts)
            print("MARKER OFFSET from truth to the detected point (m), n=%d" % len(o))
            for i, ax in enumerate("xyz"):
                print("  d%s  mean %+8.4f  sd %7.4f" % (ax, o[:, i].mean(), o[:, i].std()))
            dz = o[:, 2]
            print("  altitude span %.3f m" % span)
            if span < 0.5:
                print("  SINGLE ALTITUDE: value is well determined here (sd %.4f) but"
                      % dz.std())
                print("  unconfirmed as structural. Re-measure at another height.")
            elif dz.std() > 0.01:
                print("  NOT A FIXED OFFSET (sd %.4f over %.2f m) — suspect the camera"
                      % (dz.std(), span))
                print("  extrinsics or intrinsics, not marker_dz.")
            else:
                print("  CONFIRMED STRUCTURAL (sd %.4f over %.2f m)" % (dz.std(), span))
            print()
            print("    config/%s.yaml:  target: {marker_dz: %.4f}" % (a.site, dz.mean()))
            delta = dz.mean() - site.marker_dz
            print("    current %.4f  ->  %s%+.1f mm altitude bias"
                  % (site.marker_dz, "" if abs(delta) > 0.01 else "agrees, ",
                     delta * 1000))

    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
