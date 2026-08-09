#!/usr/bin/env bash
# ============================================================================
#  pi.sh — drive a Raspberry Pi from this machine
#
#  The board is headless and the repository is private, so the Pi has no GitHub
#  credentials. This ships the committed tree over SSH instead, runs a command in
#  the Pi's virtualenv, and copies results back.
#
#    ./bench/pi.sh setup                 create the venv and install deps (once)
#    ./bench/pi.sh sync                  push the current committed tree
#    ./bench/pi.sh run <cmd...>          run inside the venv, on the Pi
#    ./bench/pi.sh bench                 the full sweep, results pulled back
#    ./bench/pi.sh pull [subdir]         copy bench/out back to this machine
#    ./bench/pi.sh info                  board, cores, CPU features, temperature
#    ./bench/pi.sh shell                 interactive session
#
#  HOST defaults to the `ryaanpi` entry in ~/.ssh/config. Override:  HOST=other ./bench/pi.sh info
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-ryaanpi}"
DIR="${DIR:-dronevision-ai}"
PY="\$HOME/$DIR/.venv/bin/python"
BRANCH="${BRANCH:-dev}"

say() { printf '\033[1m>> %s\033[0m\n' "$*"; }

remote() { ssh "$HOST" "$@"; }

# Run inside the checkout with the venv's interpreter.
in_venv() { ssh "$HOST" "cd ~/$DIR && $PY $*"; }

case "${1:-help}" in

  info)
    remote 'echo "board   : $(tr -d "\0" </proc/device-tree/model 2>/dev/null)"
            echo "arch    : $(uname -m)   cores: $(nproc)"
            echo "features: $(grep -m1 ^Features /proc/cpuinfo | cut -d: -f2)"
            echo "dotprod : $(grep -qm1 asimddp /proc/cpuinfo && echo YES || echo "NO  (Armv8.0 - int8 gains will be limited)")"
            echo "temp    : $(awk "{printf \"%.1f C\", \$1/1000}" /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
            echo "mhz     : $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)/1000))"
            echo "disk    : $(df -h / | awk "NR==2{print \$4\" free (\"\$5\" used)\"}")"'
    ;;

  sync)
    say "shipping the committed $BRANCH tree to $HOST:~/$DIR"
    # git archive, not rsync or scp -r: it sends exactly what is committed, so a Pi
    # result can always be traced to a revision rather than to whatever was lying in
    # the working directory at the time.
    git archive --format=tar "$BRANCH" \
      | ssh "$HOST" "mkdir -p ~/$DIR && tar -x -C ~/$DIR"
    remote "du -sh ~/$DIR | cut -f1 | xargs -I{} echo '   {} on the board'"
    ;;

  setup)
    "$0" sync
    say "creating the virtualenv and installing dependencies"
    remote "cd ~/$DIR && python3 -m venv .venv 2>/dev/null; \
            .venv/bin/pip install -q --upgrade pip && \
            .venv/bin/pip install -e '.[net,ort,dev]' 2>&1 | tail -3"
    say "verifying"
    in_venv "-c 'import onnxruntime, cv2, numpy; \
                 print(\"  onnxruntime\", onnxruntime.__version__); \
                 print(\"  opencv     \", cv2.__version__); \
                 print(\"  numpy      \", numpy.__version__)'"
    ;;

  run)
    shift
    in_venv "$@"
    ;;

  bench)
    "$0" sync
    say "sweeping (one process per row: thread settings are process-global)"
    # 1 thread and all-cores are both worth having: single-thread isolates the kernel,
    # all-cores is what a deployment actually gets.
    for t in 1 4; do
      for m in models/drone_yolo26n_v4.onnx; do
        [ -z "$m" ] && continue
        name="$(basename "$m" .onnx)"
        say "  onnx  $name  threads=$t"
        in_venv "-m bench.speed --runtime onnx --model $m --threads $t \
                 --iters 60 --json bench/out/pi4-onnx-\${name}-t$t.json" || true
      done
    done
    "$0" pull
    ;;

  pull)
    sub="${2:-}"
    say "copying results back"
    mkdir -p bench/out
    # tar over ssh: scp -r trips over the Windows path handling in Git Bash.
    ssh "$HOST" "cd ~/$DIR && tar -c bench/out/$sub 2>/dev/null" | tar -x -f - || {
      echo "   (nothing to pull yet)"; exit 0; }
    ls -1 bench/out/ | sed 's/^/   /'
    ;;

  shell)
    ssh -t "$HOST" "cd ~/$DIR && exec bash -l"
    ;;

  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
