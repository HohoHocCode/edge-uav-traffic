#!/system/bin/sh
# Does this board's power rail respond to compute load?
#
#   adb push 4-bench/probe_power_android.sh /data/local/tmp/sky/
#   adb shell sh /data/local/tmp/sky/probe_power_android.sh
#
# "Performance per watt" is the entire argument for an NPU, and a benchmark
# without an energy column cannot make it. But a rail that merely *exists* is
# not a rail that *measures*: on a mains-powered board the battery gauge often
# reports a trickle that has nothing to do with SoC load, and quoting it would
# be worse than admitting there is no counter.
#
# So this does not report power. It reports whether power can be reported:
# sample idle, sample under a known load, and see whether the reading moves by
# more than its own noise. Only if it does is an energy column defensible.

SAMPLES=${SAMPLES:-25}
INTERVAL_MS=${INTERVAL_MS:-200}
PSY=/sys/class/power_supply

read_uw() {  # instantaneous power in microwatts, or empty
  for p in "$PSY"/battery "$PSY"/bms; do
    [ -d "$p" ] || continue
    pw=$(cat "$p/power_now" 2>/dev/null)
    if [ -n "$pw" ] && [ "$pw" != "0" ]; then
      echo "${pw#-}"; return
    fi
    v=$(cat "$p/voltage_now" 2>/dev/null)
    i=$(cat "$p/current_now" 2>/dev/null)
    if [ -n "$v" ] && [ -n "$i" ]; then
      # uV * uA = 1e-12 W. Divide down before multiplying: the shell is
      # 32-bit and 4.3e6 * 3.3e5 overflows instantly.
      mv=$((v / 1000))            # millivolts
      ma=${i#-}; ma=$((ma / 1000))  # milliamps, magnitude
      echo $((mv * ma))           # microwatts
      return
    fi
  done
  echo ""
}

sample_avg() {  # sample_avg <n>; echoes mean microwatts, or NONE
  n=$1; sum=0; got=0; mn=999999999; mx=0
  i=0
  while [ "$i" -lt "$n" ]; do
    u=$(read_uw)
    if [ -n "$u" ]; then
      sum=$((sum + u)); got=$((got + 1))
      [ "$u" -lt "$mn" ] && mn=$u
      [ "$u" -gt "$mx" ] && mx=$u
    fi
    i=$((i + 1))
    sleep 0.2
  done
  [ "$got" -eq 0 ] && { echo "NONE"; return; }
  echo "$((sum / got)) $mn $mx"
}

cpu_temp() {
  for z in /sys/class/thermal/thermal_zone*/; do
    t=$(cat "$z/type" 2>/dev/null)
    case "$t" in cpuss-0|cpu-1-0) cat "$z/temp" 2>/dev/null; return ;; esac
  done
  echo ""
}

echo "=== power-rail responsiveness probe ==="
echo "  rails: $(ls $PSY 2>/dev/null | tr '\n' ' ')"
u=$(read_uw)
[ -z "$u" ] && { echo "  no readable voltage/current -- no energy column is possible"; exit 1; }
echo "  reading: $((u / 1000)) mW at rest"
echo "  temp   : $(( $(cpu_temp) / 1000 )) C"
echo ""

echo "  [1/3] idle, $SAMPLES samples..."
set -- $(sample_avg "$SAMPLES"); idle=$1; idle_mn=$2; idle_mx=$3

echo "  [2/3] busy (4 shell spinners), $SAMPLES samples..."
i=0
while [ "$i" -lt 4 ]; do
  ( while :; do :; done ) &
  i=$((i + 1))
done
sleep 1
set -- $(sample_avg "$SAMPLES"); busy=$1; busy_mn=$2; busy_mx=$3
kill %1 %2 %3 %4 2>/dev/null
wait 2>/dev/null

echo "  [3/3] idle again, $SAMPLES samples..."
sleep 2
set -- $(sample_avg "$SAMPLES"); idle2=$1

echo ""
echo "  idle   : $((idle / 1000)) mW   (range $((idle_mn / 1000))-$((idle_mx / 1000)))"
echo "  busy   : $((busy / 1000)) mW   (range $((busy_mn / 1000))-$((busy_mx / 1000)))"
echo "  idle2  : $((idle2 / 1000)) mW"
echo "  temp   : $(( $(cpu_temp) / 1000 )) C"
echo ""

delta=$((busy - idle))
noise=$((idle_mx - idle_mn))
[ "$delta" -lt 0 ] && delta=$((-delta))

echo "  load delta : $((delta / 1000)) mW"
echo "  idle noise : $((noise / 1000)) mW"
if [ "$noise" -gt 0 ] && [ "$delta" -gt $((noise * 2)) ]; then
  echo ""
  echo "  VERDICT: the rail tracks compute load (delta is >2x the idle spread)."
  echo "           An energy column is defensible. Sample it alongside each"
  echo "           benchmark run and report mW and mJ/frame."
else
  echo ""
  echo "  VERDICT: the rail does NOT clearly track load. Most likely the board"
  echo "           runs from mains and the gauge reports a trickle unrelated to"
  echo "           the SoC. Do not quote watts. Report thermal and frequency"
  echo "           residency as the proxy, and say the board exposes no usable"
  echo "           energy counter."
fi
