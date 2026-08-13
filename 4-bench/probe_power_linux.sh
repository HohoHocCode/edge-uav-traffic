#!/usr/bin/env bash
# Decide whether this board can support an energy column at all.
#
# The question is not "is there a power rail" -- there usually is. It is
# whether the rail *moves* when the chip works, by more than it moves when it
# does nothing. On the Android image of this same silicon the answer was no:
# a known load shifted the battery rail by 4 mW against 26 mW of idle noise,
# so every joule figure derived from it would have been noise dressed as data.
#
# So: measure the idle distribution first, then the same statistic under load,
# and report the delta in units of the idle standard deviation. A verdict of
# "usable" means the load is separable from doing nothing; anything else means
# the column should not be published.
#
#   bash probe_power_linux.sh [seconds] [load_seconds]

set -u
N=${1:-20}          # idle samples (~10 Hz)
L=${2:-20}          # load duration
PS=/sys/class/power_supply/battery

read_mw() {
    # current_now is uA and negative while discharging; voltage_now is uV.
    local i v
    i=$(cat $PS/current_now 2>/dev/null) || return 1
    v=$(cat $PS/voltage_now 2>/dev/null) || return 1
    awk -v i="$i" -v v="$v" 'BEGIN { printf "%.1f", (i<0?-i:i)/1e6 * v/1e6 * 1000 }'
}

stats() {  # stdin: one number per line -> "mean sd n min max"
    awk '{s+=$1; ss+=$1*$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1; n++}
         END { m=s/n; sd=sqrt(ss/n-m*m); printf "%.1f %.1f %d %.1f %.1f", m, sd, n, mn, mx }'
}

sample() {  # $1 = seconds
    local end=$(( $(date +%s) + $1 ))
    while [ "$(date +%s)" -lt "$end" ]; do read_mw; echo; sleep 0.1; done
}

if [ ! -r $PS/current_now ]; then
    echo "Khong doc duoc $PS/current_now -- board khong phoi ra dong dien."
    echo "VERDICT: khong the bao cao nang luong."
    exit 1
fi

echo "Nguon: $PS  (current_now uA, voltage_now uV)"
echo "Trang thai: $(cat $PS/status 2>/dev/null)"
echo

echo "[1/3] do luc ranh ${N}s ..."
IDLE=$(sample "$N" | stats)
set -- $IDLE; I_MEAN=$1; I_SD=$2; I_N=$3; I_MIN=$4; I_MAX=$5
printf "  ranh : %8.1f mW  sd %.1f  (n=%d, %.0f..%.0f)\n" "$I_MEAN" "$I_SD" "$I_N" "$I_MIN" "$I_MAX"

echo "[2/3] do duoi tai CPU ${L}s (tat ca $(nproc) loi) ..."
for _ in $(seq "$(nproc)"); do
    ( end=$(( $(date +%s) + L + 2 )); while [ "$(date +%s)" -lt "$end" ]; do :; done ) &
done
sleep 2                       # cho tai on dinh truoc khi lay mau
LOAD=$(sample "$L" | stats)
wait 2>/dev/null
set -- $LOAD; L_MEAN=$1; L_SD=$2; L_N=$3
printf "  tai  : %8.1f mW  sd %.1f  (n=%d)\n" "$L_MEAN" "$L_SD" "$L_N"

echo "[3/3] ket luan"
DELTA=$(awk -v a="$L_MEAN" -v b="$I_MEAN" 'BEGIN{printf "%.1f", a-b}')
SIGMA=$(awk -v d="$DELTA" -v s="$I_SD" 'BEGIN{printf "%.1f", (s>0? d/s : 0)}')
printf "  chenh lech : %+.1f mW  = %.1f x do lech chuan luc ranh\n" "$DELTA" "$SIGMA"

# 5 sigma is deliberately strict. A power column is quoted per-model and per-
# precision, so the differences it must resolve are smaller than this one.
awk -v s="$SIGMA" -v d="$DELTA" 'BEGIN {
  if (s >= 5 && d > 50)      print "  VERDICT: dung duoc -- tai tach bach ro khoi trang thai ranh."
  else if (s >= 2)           print "  VERDICT: yeu -- chi bao cao duoc chenh lech lon, khong bao cao tuyet doi."
  else                       print "  VERDICT: KHONG dung duoc -- tin hieu chim trong nhieu. Dung nhiet do lam proxy."
}'
