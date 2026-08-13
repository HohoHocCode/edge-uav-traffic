#!/system/bin/sh
# On-device QNN latency benchmark for a Qualcomm board running Android.
#
#   adb push 4-bench/bench_qnn_device.sh /data/local/tmp/sky/
#   adb shell sh /data/local/tmp/sky/bench_qnn_device.sh
#
# Why not qnn-net-run's own profiling: it writes a binary log that needs
# qnn-profile-viewer from the host SDK to read. This measures wall time
# instead, which needs nothing, and isolates steady-state inference from
# one-off cost by differencing two run lengths:
#
#     t(N) = init + N * infer        ->  infer = (t(N) - t(1)) / (N - 1)
#
# Graph composition, HTP finalization and context setup all land in `init`
# and are charged once, exactly as they are in a real application that
# creates its context at start-up and reuses it. Reporting t(1) as the
# inference time -- which a naive single-shot measurement does -- overstates
# a 6 ms model by two orders of magnitude.
#
# Each configuration is repeated and the median taken, because the first
# run after a backend switch pays page-cache and clock-ramp costs.

QNN_DIR=${QNN_DIR:-/data/local/tmp/qnn}
WORK=${WORK:-/data/local/tmp/sky}
REPEATS=${REPEATS:-3}
N_LONG=${N_LONG:-101}
N_SHORT=${N_SHORT:-21}

export LD_LIBRARY_PATH=$QNN_DIR/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN_DIR/dsp

mkdir -p "$WORK" || exit 1
INPUT=$(head -1 "$QNN_DIR/input_list.txt")
[ -f "$INPUT" ] || { echo "no input tensor at $INPUT"; exit 1; }

mklist() {  # mklist <file> <count>
  : > "$1"
  i=0
  while [ "$i" -lt "$2" ]; do echo "$INPUT" >> "$1"; i=$((i + 1)); done
}

mklist "$WORK/l1.txt" 1
mklist "$WORK/lL.txt" "$N_LONG"
mklist "$WORK/lS.txt" "$N_SHORT"

# Timing on Android's toybox shell needs care on two counts.
#
# Arithmetic is 32-bit signed here, so the obvious `date +%s%N` -- or even
# `$(date +%s) * 1000` -- overflows: epoch seconds times a thousand is about
# 1.8e12 against an int32 ceiling of 2.1e9, and the result comes back
# negative. Subtracting the seconds *first* keeps every intermediate small.
#
# `date +%N` is also zero-padded, and a leading zero makes some shells read
# it as octal, so "089" becomes an error rather than 89. Strip it.
#
# And the seconds and nanoseconds must come from ONE call. Reading them as
# two separate `date` invocations races: cross a second boundary between the
# two and the pair is inconsistent by a full second, which silently turns a
# 1000 ms interval into 121 ms. That is how the first version of this script
# reported a 48 ms inference for a model that actually runs in 6.
stamp() {  # sets T_SEC and T_NS from a single atomic reading
  set -- $(date "+%s %N")
  T_SEC=$1
  T_NS=$(echo "$2" | sed 's/^0*//')
  T_NS=${T_NS:-0}
}

run_once() {  # run_once <backend.so> <dlc> <listfile> ; echoes elapsed ms
  stamp; s_sec=$T_SEC; s_ns=$T_NS
  "$QNN_DIR/bin/qnn-net-run" \
      --backend "$QNN_DIR/lib/$1" \
      --model "$QNN_DIR/lib/libQnnModelDlc.so" \
      --dlc_path "$QNN_DIR/models/$2" \
      --input_list "$3" \
      --output_dir "$WORK/out" \
      --perf_profile burst >/dev/null 2>&1
  rc=$?
  stamp; e_sec=$T_SEC; e_ns=$T_NS
  [ $rc -ne 0 ] && { echo "FAIL"; return 1; }
  [ -f "$WORK/out/Result_0/output0.raw" ] || { echo "NOOUT"; return 1; }
  echo $(( (e_sec - s_sec) * 1000 + (e_ns - s_ns) / 1000000 ))
}

median3() {  # median of the three arguments
  for v in "$1" "$2" "$3"; do echo "$v"; done | sort -n | sed -n 2p
}

bench() {  # bench <label> <backend.so> <dlc> <n_long> <longlist>
  label=$1; backend=$2; dlc=$3; n=$4; list=$5

  probe=$(run_once "$backend" "$dlc" "$WORK/l1.txt")
  case "$probe" in
    FAIL|NOOUT)
      echo "  $label : UNSUPPORTED ($probe)"
      "$QNN_DIR/bin/qnn-net-run" --backend "$QNN_DIR/lib/$backend" \
          --model "$QNN_DIR/lib/libQnnModelDlc.so" \
          --dlc_path "$QNN_DIR/models/$dlc" \
          --input_list "$WORK/l1.txt" --output_dir "$WORK/out" 2>&1 \
          | grep -iE 'error|fail' | head -2 | sed 's/^/      /'
      return ;;
  esac

  a1=$probe
  b1=$(run_once "$backend" "$dlc" "$WORK/l1.txt")
  c1=$(run_once "$backend" "$dlc" "$WORK/l1.txt")
  t1=$(median3 "$a1" "$b1" "$c1")

  aN=$(run_once "$backend" "$dlc" "$list")
  bN=$(run_once "$backend" "$dlc" "$list")
  cN=$(run_once "$backend" "$dlc" "$list")
  tN=$(median3 "$aN" "$bN" "$cN")

  # x100 to keep one decimal place in integer arithmetic
  per=$(( (tN - t1) * 100 / (n - 1) ))
  fps=$(( 100000 / per ))
  echo "  $label : init ${t1} ms | infer $((per / 100)).$((per % 100)) ms/frame | ~${fps} fps (compute only)"
}

io_baseline() {
  # qnn-net-run re-reads the input tensor from disk and writes the output for
  # EVERY inference in the list. At 1x3x640x640 float32 that is 4.9 MB in and
  # roughly 0.5 MB out per frame, so the differenced figure above is
  # (file read + inference + file write), not inference. A real application
  # holds the tensor in memory and pays none of it.
  #
  # Measuring the I/O separately is the only way to say how much of the
  # reported number is the NPU and how much is the filesystem.
  n=100

  # Every `cat` or `dd` below is a fresh process, and process creation on
  # Android costs tens of milliseconds -- far more than reading a file that
  # is already in page cache. Timing the loop directly measures fork+exec,
  # not I/O, and produces the absurd result of "I/O" exceeding the total
  # inference time it is supposedly a part of. Measure the spawn cost with a
  # no-op loop and subtract it.
  timed_loop() {  # timed_loop <n> <command...>; echoes ms
    stamp; l_sec=$T_SEC; l_ns=$T_NS
    j=0
    while [ "$j" -lt "$1" ]; do shift_cmd="$2"; $shift_cmd >/dev/null 2>&1; j=$((j + 1)); done
    stamp
    echo $(( (T_SEC - l_sec) * 1000 + (T_NS - l_ns) / 1000000 ))
  }

  stamp; a_sec=$T_SEC; a_ns=$T_NS
  i=0; while [ "$i" -lt "$n" ]; do true; i=$((i + 1)); done
  stamp
  noop=$(( (T_SEC - a_sec) * 1000 + (T_NS - a_ns) / 1000000 ))

  stamp; a_sec=$T_SEC; a_ns=$T_NS
  i=0; while [ "$i" -lt "$n" ]; do /system/bin/true; i=$((i + 1)); done
  stamp
  spawn=$(( (T_SEC - a_sec) * 1000 + (T_NS - a_ns) / 1000000 ))

  stamp; a_sec=$T_SEC; a_ns=$T_NS
  i=0; while [ "$i" -lt "$n" ]; do cat "$INPUT" > /dev/null; i=$((i + 1)); done
  stamp
  rd=$(( (T_SEC - a_sec) * 1000 + (T_NS - a_ns) / 1000000 ))

  rd_net=$((rd - spawn))
  [ "$rd_net" -lt 0 ] && rd_net=0

  echo ""
  echo "  --- process-spawn vs file-read, $n iterations ---"
  echo "  shell loop only     : ${noop} ms"
  echo "  spawn /system/bin/true: ${spawn} ms   ($((spawn * 100 / n)) /100 ms per spawn)"
  echo "  spawn + read 4.9 MB : ${rd} ms"
  echo "  => read cost alone  : ${rd_net} ms   ($((rd_net * 100 / n)) /100 ms per frame)"
  echo ""
  echo "  The input file is in page cache after the first read, so this is a"
  echo "  memcpy, not disk. It bounds how much of the figures above is I/O:"
  echo "  small. The rest is compute plus qnn-net-run's per-inference setup."
}

echo "=== QNN on-device benchmark ==="
echo "  soc     : $(cat /sys/devices/soc0/machine 2>/dev/null)"
echo "  sdk     : $("$QNN_DIR/bin/qnn-net-run" --version 2>/dev/null | tail -1)"
echo "  repeats : $REPEATS (median), N=$N_LONG long / $N_SHORT short"
echo ""
bench "HTP  w8a8 (NPU)" libQnnHtp.so yolov8n_w8a8.dlc "$N_LONG"  "$WORK/lL.txt"
bench "HTP  fp32 (NPU)" libQnnHtp.so yolov8n_fp32.dlc "$N_LONG"  "$WORK/lL.txt"
bench "CPU  w8a8 (Kryo)" libQnnCpu.so yolov8n_w8a8.dlc "$N_SHORT" "$WORK/lS.txt"
bench "CPU  fp32 (Kryo)" libQnnCpu.so yolov8n_fp32.dlc "$N_SHORT" "$WORK/lS.txt"
bench "GPU  w8a8 (Adreno)" libQnnGpu.so yolov8n_w8a8.dlc "$N_SHORT" "$WORK/lS.txt"
bench "GPU  fp32 (Adreno)" libQnnGpu.so yolov8n_fp32.dlc "$N_SHORT" "$WORK/lS.txt"
io_baseline
echo ""
echo "note: these figures include qnn-net-run's per-frame file I/O (see the"
echo "      baseline above) and exclude camera capture, letterbox, NMS and"
echo "      tracking. Neither matches a deployed application; the frame-budget"
echo "      table is the end-to-end number."
