#!/usr/bin/env python3
"""Turn benchmark JSON rows into the results table.

Each row was produced by `bench.speed` on some machine; this collects them and renders
markdown for the README. Adding a board is one more JSON file, not a code change.

    python -m bench.report bench/out/*.json
    python -m bench.report bench/out/*.json --markdown > results.md

WHAT IT REFUSES TO DO. It will not put two rows side by side if they measured different
work - a different input resolution, a different artifact, or a different thread count
makes a ratio meaningless, and printing one anyway is how a resolution artifact becomes a
published speedup. Mismatches are reported as warnings above the table rather than
quietly averaged away.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(paths):
    rows = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            try:
                rows.append(json.loads(Path(p).read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                print(f"skipping {p}: {e}", file=sys.stderr)
    return rows


def board_of(row):
    host = row.get("host", {})
    board = host.get("board")
    if board:
        return board.replace("Raspberry Pi", "Pi").replace(" Model B", "").strip()
    return f"{host.get('hostname', '?')} ({host.get('machine', '?')})"


def core_of(row):
    host = row.get("host", {})
    if host.get("cpu_model"):
        return host["cpu_model"]
    if "has_dotprod" in host:
        return "Cortex-A76" if host["has_dotprod"] else "Cortex-A72"
    return host.get("machine", "?")


def check_comparable(rows):
    """Anything that would make two rows measure different amounts of work."""
    warnings = []
    for field, label in (("input", "input shape"), ("model_sha12", "artifact")):
        seen = {r["model_desc"].get(field) for r in rows}
        if len(seen) > 1 and field == "input":
            warnings.append(
                f"rows use different {label}s ({sorted(map(str, seen))}) - their "
                f"timings are NOT comparable; the network is doing different work")
        elif len(seen) > 1:
            warnings.append(f"rows span {len(seen)} different {label}s")
    for r in rows:
        th = r.get("thermal", {})
        b, a = th.get("mhz_before"), th.get("mhz_after")
        if b and a and a < b * 0.95:
            warnings.append(
                f"{r['label']}: CPU clock fell {b} -> {a} MHz during the run; "
                f"thermally limited, treat as a floor not a measurement")
    return warnings


def render(rows, markdown=True):
    rows = sorted(rows, key=lambda r: (board_of(r), r["model_desc"].get("precision", ""),
                                       r["model_desc"]["threads_intra"]))
    out = []
    warn = check_comparable(rows)
    for w in warn:
        out.append(f"> **warning:** {w}")
    if warn:
        out.append("")

    hdr = ["Board", "Core", "dotprod", "Runtime", "Precision", "Threads",
           "Inference ms", "Total ms", "inf/s"]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        d, h, s = r["model_desc"], r.get("host", {}), r["stages_ms"]
        dot = {True: "yes", False: "**no**"}.get(h.get("has_dotprod"), "n/a")
        lines.append("| " + " | ".join([
            board_of(r), core_of(r), dot,
            d.get("runtime", "?"), d.get("precision", "?"),
            str(d.get("threads_intra", "?")),
            f"{s['infer']['mean']:.1f}", f"{s['total']['mean']:.1f}",
            f"{r['fps']:.1f}",
        ]) + " |")
    out.extend(lines)

    # Ratios only within an identical configuration, so a comparison never straddles a
    # change in the work being done.
    groups = {}
    for r in rows:
        key = (r["model_desc"].get("runtime"), r["model_desc"].get("precision"),
               r["model_desc"]["threads_intra"])
        groups.setdefault(key, []).append(r)
    ratios = []
    for (rt, prec, th), grp in sorted(groups.items()):
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=lambda r: r["stages_ms"]["infer"]["mean"])
        fast, slow = grp[0], grp[-1]
        ratios.append(f"- `{rt}/{prec}` at {th} thread(s): **{board_of(slow)} is "
                      f"{slow['stages_ms']['infer']['mean'] / fast['stages_ms']['infer']['mean']:.1f}x "
                      f"slower** than {board_of(fast)}")
    if ratios:
        out += ["", "**Same configuration, different boards:**", ""] + ratios

    # Thread scaling, per board.
    by_board = {}
    for r in rows:
        by_board.setdefault((board_of(r), r["model_desc"].get("precision")), []).append(r)
    scal = []
    for (board, prec), grp in sorted(by_board.items()):
        ts = {r["model_desc"]["threads_intra"]: r["stages_ms"]["infer"]["mean"]
              for r in grp}
        if 1 in ts and max(ts) > 1:
            n = max(ts)
            scal.append(f"- {board} {prec}: {ts[1]:.0f} ms at 1 thread -> "
                        f"{ts[n]:.0f} ms at {n} threads = **{ts[1] / ts[n]:.2f}x**, "
                        f"against a {n}x ideal")
    if scal:
        out += ["", "**Thread scaling:**", ""] + scal

    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["bench/out/*.json"])
    ap.add_argument("--markdown", action="store_true", help="table only, for pasting")
    a = ap.parse_args(argv)

    rows = load(a.paths or ["bench/out/*.json"])
    if not rows:
        raise SystemExit("no result files; run `python -m bench.speed --json ...` first")
    if not a.markdown:
        print(f"{len(rows)} rows\n")
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
