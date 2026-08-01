#!/usr/bin/env python3
"""Measure inference cost, with enough context that the number means something.

One row of the results table. Runs the detector over real corpus frames and reports
per-stage milliseconds, alongside a full description of what was measured: which
artifact (by content hash), which engine and version, what input shape, how many threads,
and — on a Raspberry Pi — the CPU frequency and temperature before and after.

WHY ALL THAT PROVENANCE. A latency figure is meaningless on its own. Two runs differing
only in thread count, input resolution or board temperature can differ by more than the
optimization being measured, and a table that mixes them silently is worse than no table.
This tool refuses to emit a row it cannot describe.

    python -m bench.speed --runtime onnx --threads 4 --json out/pi4-onnx-fp32.json
    python -m bench.speed --runtime onnx --model models/drone_int8.onnx --threads 1

Thread-affecting settings are process-global in every backend, so measure ONE
configuration per process — run the sweep as separate invocations, not a loop.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _read(path, default=None):
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return default


def cpu_temp_c():
    """Board temperature, or None. A Pi that throttles mid-sweep produces a number that
    looks like a regression but is really a thermal limit."""
    raw = _read("/sys/class/thermal/thermal_zone0/temp")
    if raw and raw.isdigit():
        return round(int(raw) / 1000.0, 1)
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                             text=True, timeout=3).stdout
        return float(out.split("=")[1].split("'")[0])
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def cpu_mhz():
    raw = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return round(int(raw) / 1000.0) if raw and raw.isdigit() else None


def host_info():
    """Everything about the machine that could change a timing."""
    info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "board": _read("/proc/device-tree/model"),
        "temp_c": cpu_temp_c(),
        "mhz": cpu_mhz(),
    }
    cpuinfo = _read("/proc/cpuinfo", "") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("Features") and "cpu_features" not in info:
            info["cpu_features"] = line.split(":", 1)[1].strip()
        if line.startswith("model name") and "cpu_model" not in info:
            info["cpu_model"] = line.split(":", 1)[1].strip()
    # The flag the whole project turns on: int8 dot-product acceleration exists only on
    # Armv8.2+ cores. Recorded per row so a Pi 4 and a Pi 5 result are never confused.
    feats = info.get("cpu_features", "")
    if feats:
        info["has_dotprod"] = "asimddp" in feats
        info["has_i8mm"] = "i8mm" in feats
    return info


def summarize(samples_ms):
    s = sorted(samples_ms)
    n = len(s)
    return {
        "n": n,
        "mean": round(statistics.fmean(s), 4),
        "median": round(s[n // 2], 4),
        "p95": round(s[min(n - 1, int(0.95 * n))], 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "stdev": round(statistics.pstdev(s), 4) if n > 1 else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--runtime", default="onnx")
    ap.add_argument("--model", default=None, help="artifact; defaults per runtime")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--threads", type=int, default=None,
                    help="REQUIRED for a comparable row; unset means all cores")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--json", default=None, help="write the row here")
    ap.add_argument("--label", default=None, help="row label; default <board>-<rt>-<prec>")
    a = ap.parse_args(argv)

    from dronevision.io.sources.replay import ReplaySource
    from dronevision.l2_perception.detector import YoloDetector

    host = host_info()
    print(f"host   {host.get('board') or host['hostname']}  {host['machine']}  "
          f"{host['cpu_count']} cores"
          + (f"  dotprod={host['has_dotprod']}" if "has_dotprod" in host else ""))

    det = YoloDetector(model=a.model, imgsz=a.imgsz, runtime=a.runtime,
                       threads=a.threads, conf=a.conf)
    desc = det.describe()
    print(f"model  {desc['model']}  sha {desc['model_sha12']}  "
          f"{desc.get('precision', '?')}  {desc.get('input', '?')}")
    print(f"engine {desc['runtime']}  threads intra={desc['threads_intra']} "
          f"opencv={desc['threads_opencv']}  layout={desc.get('layout')}")
    print()

    # Real frames, cycled. Synthetic noise would exercise different branches in the
    # postprocess (nothing above threshold) and is not what the deployment sees.
    src = ReplaySource(a.corpus, mode="decoded", limit=max(20, a.iters // 4))
    frames = []
    for st in src:
        for cam in st.cams:
            img = st.latest(cam)
            if img is not None:
                frames.append(img)
    if not frames:
        raise SystemExit(f"no frames in {a.corpus}")

    for i in range(a.warmup):
        det.runtime.detect_boxes(frames[i % len(frames)], conf=a.conf, max_det=1)

    temp_before, mhz_before = cpu_temp_c(), cpu_mhz()
    pre, inf, post, total, hits = [], [], [], [], 0
    t_wall = time.perf_counter()
    for i in range(a.iters):
        t0 = time.perf_counter()
        d = det.runtime.detect_boxes(frames[i % len(frames)], conf=a.conf, max_det=1)
        total.append((time.perf_counter() - t0) * 1e3)
        t = det.runtime.last_timings
        pre.append(t["pre_ms"]); inf.append(t["infer_ms"]); post.append(t["post_ms"])
        hits += len(d) > 0
    wall = time.perf_counter() - t_wall
    temp_after, mhz_after = cpu_temp_c(), cpu_mhz()

    row = {
        "label": a.label or "-".join(filter(None, [
            (host.get("board") or host["hostname"]).split()[-1].lower(),
            desc["runtime"], desc.get("precision", "fp32"),
            f"t{desc['threads_intra']}"])),
        "stages_ms": {"pre": summarize(pre), "infer": summarize(inf),
                      "post": summarize(post), "total": summarize(total)},
        "fps": round(a.iters / wall, 2),
        "detection_rate": round(hits / a.iters, 3),
        "iters": a.iters,
        "warmup": a.warmup,
        "frames_used": len(frames),
        "model_desc": desc,
        "host": host,
        "thermal": {"temp_before_c": temp_before, "temp_after_c": temp_after,
                    "mhz_before": mhz_before, "mhz_after": mhz_after},
    }

    st = row["stages_ms"]
    print("STAGE (ms)      mean   median      p95      max")
    for k in ("pre", "infer", "post", "total"):
        s = st[k]
        print("  %-10s %7.3f %8.3f %8.3f %8.3f"
              % (k, s["mean"], s["median"], s["p95"], s["max"]))
    print()
    print(f"  {row['fps']:.1f} inferences/s sustained   "
          f"detection rate {row['detection_rate']:.0%}")

    if temp_before and temp_after:
        drift = temp_after - temp_before
        print(f"  temperature {temp_before:.1f} -> {temp_after:.1f} C")
        if mhz_before and mhz_after and mhz_after < mhz_before * 0.95:
            print(f"  !! CPU CLOCK DROPPED {mhz_before} -> {mhz_after} MHz — this row is "
                  f"thermally limited and is not comparable to a cool one")
        elif drift > 12:
            print(f"  !! warmed {drift:.0f} C during the run; re-run cool to confirm")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
