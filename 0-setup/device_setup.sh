#!/usr/bin/env bash
# On-board setup and capability probe for a Qualcomm edge device.
#
# Run once after deploy.ps1:   bash device_setup.sh
#
# It is deliberately tolerant: Qualcomm Linux, Ubuntu-on-Android-kernel and
# Yocto images differ in what they ship, so every step reports what happened
# and continues rather than aborting the whole script.

set -u

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'

say()  { echo "${BOLD}==> $*${RESET}"; }
ok()   { echo "  ${GREEN}ok${RESET}   $*"; }
warn() { echo "  ${YELLOW}warn${RESET} $*"; }
bad()  { echo "  ${RED}fail${RESET} $*"; }

say "System"
echo "  $(uname -a)"
[ -f /etc/os-release ] && . /etc/os-release && echo "  distro: ${PRETTY_NAME:-unknown}"
echo "  cpus:   $(nproc 2>/dev/null || echo '?')"
echo "  mem:    $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"

say "SoC identification"
for f in /sys/devices/soc0/machine /sys/devices/soc0/family /sys/devices/soc0/soc_id; do
  [ -r "$f" ] && echo "  $(basename "$f"): $(cat "$f" 2>/dev/null)"
done

say "Python"
if command -v python3 >/dev/null 2>&1; then
  ok "$(python3 -V 2>&1)"
else
  bad "python3 missing - install it before continuing"
  exit 1
fi

say "Python packages"
PKGS_OK=1
python3 - <<'PY' || PKGS_OK=0
import importlib, sys
need = {"numpy": None, "cv2": "opencv-python-headless", "yaml": "pyyaml",
        "onnxruntime": "onnxruntime"}
missing = []
for mod, pipname in need.items():
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:14s} {getattr(m, '__version__', '')}")
    except ImportError:
        missing.append(pipname or mod)
        print(f"  MISS {mod}")
if missing:
    print("  install with: pip3 install " + " ".join(missing))
    sys.exit(1)
PY

if [ "$PKGS_OK" -eq 0 ]; then
  warn "attempting install of the missing packages"
  pip3 install --no-cache-dir numpy opencv-python-headless pyyaml onnxruntime 2>&1 | tail -3 \
    || warn "pip install failed - you may need a proxy or an offline wheel"
fi

say "ONNX Runtime execution providers"
python3 - <<'PY'
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    print(f"  onnxruntime {ort.__version__}")
    print(f"  providers: {provs}")
    if "QNNExecutionProvider" in provs:
        print("  ok   QNN EP present - the NPU path is available")
    else:
        print("  warn QNN EP absent - inference will run on the Kryo CPU.")
        print("       For the NPU: pip3 install onnxruntime-qnn")
        print("       (aarch64 wheel; needs the QAIRT libs on LD_LIBRARY_PATH)")
except ImportError:
    print("  onnxruntime not installed")
PY

say "QAIRT / QNN libraries"
FOUND=0
for d in /usr/lib /usr/local/lib /opt/qcom /opt/qti-aic /data/local/tmp; do
  [ -d "$d" ] || continue
  hits=$(find "$d" -maxdepth 3 -name 'libQnn*.so' 2>/dev/null | head -6)
  if [ -n "$hits" ]; then
    FOUND=1
    echo "  in $d:"
    echo "$hits" | sed 's/^/    /'
  fi
done
[ "$FOUND" -eq 0 ] && warn "no libQnn*.so found - the HTP backend will not load"

say "Camera devices"
ls /dev/video* 2>/dev/null | sed 's/^/  /' || warn "no /dev/video* nodes"

say "Power / thermal telemetry"
if [ -f "4-bench/probe_power.py" ]; then
  python3 4-bench/probe_power.py --discover 2>&1 | sed 's/^/  /'
else
  warn "4-bench/probe_power.py not deployed"
fi

say "Model artefacts"
if ls models/*.onnx >/dev/null 2>&1; then
  for m in models/*.onnx; do
    echo "  $(ls -lh "$m" | awk '{print $9, $5}')"
  done
else
  warn "no models/*.onnx on the device"
fi

say "Done"
cat <<'EOF'
  Next:
    python3 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx --iters 50
    python3 3-pipeline/run_pipeline.py --source 0 --headless --max-frames 300
EOF
