#!/system/bin/sh
# Time an AI Hub QNN context binary on the board itself.
#
# Why this is not just `time qnn-net-run`:
#
#   A single run measures process spawn + backend load + context
#   deserialisation + graph finalisation + N inferences + output writes.
#   On this board the fixed part is ~200 ms and the inference is ~5 ms, so a
#   single-run number is 97 % overhead. Three earlier attempts at this
#   produced three confidently wrong figures.
#
# So: run the same binary twice, once with N1 inferences and once with N2, and
# difference them. Everything constant cancels, including the process spawn:
#
#       infer = (t(N2) - t(N1)) / (N2 - N1)
#
# --keep_num_outputs=1 stops it writing a 470 KB tensor per iteration, which
# would otherwise be measured as "inference".
#
# Timestamps come from ONE `date` call reading both fields. Calling
# `date +%s` and `date +%N` separately races across the second boundary and
# silently reports ~1000 ms errors. `%s%N` alone overflows this shell's 32-bit
# arithmetic.
#
#   sh bench_ctx_device.sh <context.bin> <input_list.txt> [tag]

set -e

QNN=/data/local/tmp/qnn
CTX=${1:?usage: bench_ctx_device.sh <context.bin> <input_list.txt> [tag]}
LIST=${2:?missing input list}
TAG=${3:-$(basename "$CTX" .bin)}

N1=5
N2=105
PERF=${4:-burst}

export LD_LIBRARY_PATH=$QNN/lib
export ADSP_LIBRARY_PATH=$QNN/dsp

stamp() {
    # One call, two fields. Prints milliseconds since epoch.
    set -- $(date "+%s %N")
    echo $(( $1 * 1000 + $2 / 1000000 ))
}

run() {
    n=$1
    "$QNN/bin/qnn-net-run" \
        --retrieve_context "$CTX" \
        --backend "$QNN/lib/libQnnHtp.so" \
        --input_list "$LIST" \
        --output_dir /data/local/tmp/sky/ctx/_t \
        --num_inferences "$n" \
        --keep_num_outputs 1 \
        --perf_profile "$PERF" >/dev/null 2>&1
}

# Warm up: first load pays for DSP session setup and page-in, and charging
# that to inference is how a 5 ms model gets reported as 200 ms.
run $N1

t0=$(stamp); run $N1;  t1=$(stamp)
t2=$(stamp); run $N2;  t3=$(stamp)

a=$(( t1 - t0 ))
b=$(( t3 - t2 ))
d=$(( b - a ))
n=$(( N2 - N1 ))

# Integer shell: scale before dividing to keep three decimals.
us=$(( d * 1000 / n ))

echo "tag           $TAG"
echo "context       $CTX"
echo "perf_profile  $PERF"
echo "t($N1)          ${a} ms"
echo "t($N2)         ${b} ms"
echo "delta         ${d} ms over ${n} inferences"
echo "inference     ${us} us"
echo "fixed_cost    $(( a - (N1 * us / 1000) )) ms   (spawn + load + finalise)"
