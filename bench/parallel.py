#!/usr/bin/env python3
"""Where should the parallelism go: inside one inference, or across cameras?

The pipeline must detect on four cameras per fix. A fixed number of cores can be spent
two ways, and they are not equivalent:

    sequential   one camera at a time, all cores inside each inference
    parallel     one camera per core, a single-threaded engine in each

Inference on this network is memory-bound, so splitting one image across cores returns
well under linear. Four separate images have no such problem. This measures the whole
trade rather than assuming it, and reports the per-fix latency that actually matters —
four cameras localized, not one inference finished.

    python -m bench.parallel                       # sweep every split
    python -m bench.parallel --json bench/out/pi4-parallel.json
    python -m bench.parallel --detector color      # is it worth it for a cheap detector?

Run on an otherwise idle machine. Every configuration competes for the same cores, so a
background process distorts the comparison rather than adding noise to it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bench.speed import cpu_mhz, cpu_temp_c, host_info, summarize


def load_framesets(corpus, cams, n):
    """Real frame sets: a fix needs all cameras of one instant, not n loose images."""
    from dronevision.io.sources.replay import ReplaySource

    src = ReplaySource(corpus, mode="decoded", limit=n)
    sets = []
    for st in src:
        fs = {c: st.latest(c) for c in cams}
        if all(v is not None for v in fs.values()):
            sets.append(fs)
    if not sets:
        raise SystemExit(f"no complete frame sets in {corpus}")
    return sets


def time_strategy(detect_fn, framesets, iters, warmup):
    for i in range(warmup):
        detect_fn(framesets[i % len(framesets)])
    per_fix, found = [], []
    for i in range(iters):
        fs = framesets[i % len(framesets)]
        t0 = time.perf_counter()
        dets = detect_fn(fs)
        per_fix.append((time.perf_counter() - t0) * 1e3)
        found.append(len(dets))
    return per_fix, found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--runtime", default="onnx")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cores", type=int, default=None, help="default: all")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from dronevision.l2_perception.detector import make_detector
    from dronevision.l2_perception.parallel import ParallelPerception
    from dronevision.l4_triangulation.calibration import load_site

    site = load_site(a.site)
    cams = site.cam_names
    cores = a.cores or os.cpu_count() or 1
    host = host_info()

    print(f"host    {host.get('board') or host['hostname']}  {host['machine']}  "
          f"{cores} cores"
          + (f"  dotprod={host['has_dotprod']}" if "has_dotprod" in host else ""))
    print(f"work    {len(cams)} cameras per fix  |  {a.detector}/{a.runtime} @{a.imgsz}")
    print(f"iters   {a.iters} fixes ({a.iters * len(cams)} inferences), "
          f"{a.warmup} warmup\n")

    framesets = load_framesets(a.corpus, cams, max(12, a.iters))

    det_kw = {}
    if a.detector in ("yolo", "hybrid"):
        det_kw = {"runtime": a.runtime, "imgsz": a.imgsz}

    # Splits worth measuring: all cores on one image at a time, through to one core per
    # camera. Intermediate splits catch the case where neither extreme is best.
    configs = []
    for workers in sorted({1, 2, len(cams), cores}):
        if workers < 1 or workers > len(cams):
            continue
        tpw = max(1, cores // workers)
        configs.append((workers, tpw))

    rows = []
    for workers, tpw in configs:
        label = ("sequential" if workers == 1 else f"parallel x{workers}")
        print(f"  {label:<16} {workers} worker(s) x {tpw} thread(s) "
              f"= {workers * tpw} of {cores} cores ... ", end="", flush=True)

        temp0, mhz0 = cpu_temp_c(), cpu_mhz()
        if workers == 1:
            det = make_detector(a.detector, threads=tpw, **det_kw)

            def run(fs, _d=det):
                out = {}
                for c, img in fs.items():
                    uv = _d.detect(img, cam=c)
                    if uv:
                        out[c] = uv
                return out
            closer = getattr(getattr(det, "runtime", None), "close", lambda: None)
        else:
            par = ParallelPerception(
                cams,
                lambda cam, th: make_detector(a.detector,
                                              **({"threads": th, **det_kw}
                                                 if det_kw else {})),
                workers=workers, threads_per_worker=tpw)
            run = par.detect_all
            closer = par.close

        per_fix, found = time_strategy(run, framesets, a.iters, a.warmup)
        closer()
        temp1, mhz1 = cpu_temp_c(), cpu_mhz()

        s = summarize(per_fix)
        rows.append({
            "label": label, "workers": workers, "threads_per_worker": tpw,
            "cores_used": workers * tpw,
            "ms_per_fix": s, "fix_hz": round(1000.0 / s["mean"], 2),
            "ms_per_camera": round(s["mean"] / len(cams), 2),
            "detections_per_fix": round(float(np.mean(found)), 2),
            "thermal": {"temp_before_c": temp0, "temp_after_c": temp1,
                        "mhz_before": mhz0, "mhz_after": mhz1},
        })
        print(f"{s['mean']:7.1f} ms/fix   {rows[-1]['fix_hz']:5.2f} Hz")

    print()
    print("%-18s %8s %10s %8s %9s %8s" % ("strategy", "cores", "ms/fix", "Hz",
                                          "ms/cam", "vs seq"))
    base = next((r for r in rows if r["workers"] == 1), None)
    for r in rows:
        gain = (base["ms_per_fix"]["mean"] / r["ms_per_fix"]["mean"]) if base else 1.0
        print("%-18s %8d %10.1f %8.2f %9.1f %7.2fx"
              % (f"{r['workers']}x{r['threads_per_worker']}t", r["cores_used"],
                 r["ms_per_fix"]["mean"], r["fix_hz"], r["ms_per_camera"], gain))

    best = min(rows, key=lambda r: r["ms_per_fix"]["mean"])
    print()
    if base and best is not base:
        print(f"  BEST: {best['workers']} workers x {best['threads_per_worker']} thread(s)"
              f"  ->  {base['ms_per_fix']['mean'] / best['ms_per_fix']['mean']:.2f}x "
              f"faster per fix than all-cores-on-one-image")
        print(f"        {base['fix_hz']:.2f} Hz -> {best['fix_hz']:.2f} Hz")
    else:
        print("  No split beat the sequential baseline on this machine/detector.")

    for r in rows:
        th = r["thermal"]
        if th["mhz_before"] and th["mhz_after"] and \
                th["mhz_after"] < th["mhz_before"] * 0.95:
            print(f"  !! {r['label']}: clock fell {th['mhz_before']} -> "
                  f"{th['mhz_after']} MHz; thermally limited, treat as a floor")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"experiment": "camera-parallelism", "host": host, "cores": cores,
             "cameras": cams, "detector": a.detector, "runtime": a.runtime,
             "imgsz": a.imgsz, "iters": a.iters, "rows": rows},
            indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
