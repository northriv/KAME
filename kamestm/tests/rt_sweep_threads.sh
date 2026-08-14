#!/bin/bash
# Copyright (C) 2002-2026 Kentaro Kitagawa
#
# T-sweep for the HIGHEST rebuild bound, for a PREEMPT_RT host.
#
# WHAT IT TESTS.  NegotiateReserve.tla derives a per-Linkage bound on the
# expensive rebuild count -- the quantity Node::snapshot reports as
# snapshot_retries_max -- of the form
#
#     rebuilds <= (T-1) * K * L
#
# with T the thread count, K the CASes a peer may land per negotiate (2 today:
# bundle Phase 2 and Phase 4 share one scope), and L the transaction's own
# tagged-Linkage count.  The T in that formula is the whole point: TLC showed a
# thread-count-INDEPENDENT constant -- which is what the "3" in the original 3L
# rule was -- survives only the single-peer case.  A four-thread box cannot
# test that.  Sweeping T is what an RT machine with cores is for.
#
# The sweep runs one build against several T and asks whether the measured
# maximum tracks (T-1)*K*L.  Two outcomes are informative:
#   * it tracks    -> the model holds on real hardware; the bound is usable.
#   * it does not  -> there is a path the single-Linkage model does not see,
#                     and the next step is to find it rather than to publish a
#                     bound.  Candidates, in the order worth checking: multiple
#                     bundle levels each contributing independently; rebuilds
#                     arriving from Node::snapshot's self-promote CAS or its
#                     weak-acquire loss rather than from bundle(); L itself
#                     growing with T.
#
# PINNING.  KAME_MIX_OS_PIN=1 puts acquisition alone on the last CPU and
# EVERYONE ELSE ON CPU 0 -- the isolcpus shape KAME deploys.  Peers then
# time-share one core, so raising T does not raise their concurrency, only
# their preemption.  That is the shipped shape and worth measuring, but it is
# not the shape that tests a bound whose parameter is the number of peers that
# can be in flight at once.  The sweep therefore runs UNPINNED by default and
# takes PIN=1 for the deployment-shape confirmation.
#
# DISCIPLINE, learned the hard way on this branch:
#   * One binary per arm, checked with cmp before use -- a header edit that the
#     build did not pick up produces a confident comparison of a binary with
#     itself.
#   * Never judge a maximum on a short run.  Sixty seconds does not reach the
#     tail; twelve seconds swings wildly.  15 minutes minimum, and n=1 proves
#     nothing (a single 30-minute soak on this branch read "over by one in 115 M
#     commits" and four repeats put the excursion at +9 to +26).
#   * Report the maximum WITH the L it was measured against.  They come from
#     different counters and a bound compared against the wrong L is not a
#     result.
#   * Ascending then descending, so drift across the job separates from T.
#   * Record the environment.  A run whose provenance is not in the log cannot
#     be compared with anything later.
#
# Usage:
#   tests/rt_sweep_threads.sh <path-to-transaction_priority_mixed_test> [outdir]
# Environment:
#   SECS=900     seconds per run
#   TLIST="3 4 6 8"
#   PIN=0        1 = KAME_MIX_OS_PIN (deployment shape; peers share CPU 0)
#   FIFO=0       KAME_MIX_OS_FIFO priority for acquisition, 0 = off
#   REPS=1       ascending+descending passes

set -u

BIN=${1:?usage: rt_sweep_threads.sh <binary> [outdir]}
OUT=${2:-rt_sweep_$$}
SECS=${SECS:-900}
TLIST=${TLIST:-"3 4 6 8"}
PIN=${PIN:-0}
FIFO=${FIFO:-0}
REPS=${REPS:-1}

[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 1; }
mkdir -p "$OUT" || exit 1

# ---- provenance -------------------------------------------------------------
{
    echo "date          $(date -Is)"
    echo "host          $(uname -n)"
    echo "kernel        $(uname -a)"
    echo "nproc         $(nproc)"
    echo "clocksource   $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null)"
    echo "governor      $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "binary        $BIN"
    echo "binary sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "git rev       $(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null)"
    echo "git dirty     $(git -C "$(dirname "$0")" status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "SECS=$SECS TLIST='$TLIST' PIN=$PIN FIFO=$FIFO REPS=$REPS"
} | tee "$OUT/provenance.txt"
echo

nprocs=$(nproc)
for T in $TLIST; do
    [ "$T" -le "$nprocs" ] || echo "NOTE: T=$T exceeds nproc=$nprocs — peers will time-share regardless of PIN"
done

# T = 1 (acq) + 1 (UI) + scripting + normals.  Split the remainder evenly, with
# the odd one going to NORMAL (the tier that shares the acquiring subtree).
run_one() {   # $1 = T, $2 = tag
    local T=$1 tag=$2 rest scr nrm log
    rest=$((T - 2))
    [ "$rest" -lt 1 ] && rest=1
    scr=$((rest / 2)); nrm=$((rest - scr))
    log="$OUT/T${T}.${tag}.log"
    KAME_MIX_SECS=$SECS \
    KAME_MIX_LEAVES=4 \
    KAME_MIX_OS_PIN=$PIN \
    KAME_MIX_OS_FIFO=$FIFO \
    KAME_MIX_SCRIPTING=$scr \
    KAME_MIX_NORMALS=$nrm \
        "$BIN" > "$log" 2>&1
    echo "T=$T ($tag) scripting=$scr normals=$nrm exit=$? -> $log"
}

for r in $(seq 1 "$REPS"); do
    for T in $TLIST;                     do run_one "$T" "up$r";   done
    for T in $(echo $TLIST | tr ' ' '\n' | tac | tr '\n' ' '); do
                                             run_one "$T" "down$r"; done
done

# ---- report -----------------------------------------------------------------
echo
printf "%-5s %-7s %-5s %-4s %-9s %-10s %-11s %-9s %-9s %s\n" \
       T pass reb L "(T-1)KL" "margin" "acq/s" p99.9 p99.999 verdict
for f in "$OUT"/T*.log; do
    T=$(basename "$f" | sed 's/^T\([0-9]*\)\..*/\1/')
    pass=$(basename "$f" | sed 's/^T[0-9]*\.\(.*\)\.log/\1/')
    reb=$(grep -a "rebuilds: max=" "$f" | grep -oE "max=[0-9]+" | head -1 | cut -d= -f2)
    L=$(grep -a "rebuilds: max=" "$f" | grep -oE "L=[0-9]+" | head -1 | cut -d= -f2)
    acq=$(grep -a "acq(record)" "$f" | grep -oE "\([0-9]+ /s" | tr -d '( /s')
    p999=$(grep -a "n=.*mean=" "$f" | grep -oE "p99\.9=[0-9]+" | head -1 | cut -d= -f2)
    p5=$(grep -a "n=.*mean=" "$f" | grep -oE "p99\.999=[0-9]+" | head -1 | cut -d= -f2)
    v=$(grep -aoE "^PASSED|^FAILED|STALL" "$f" | head -1)
    if [ -n "$reb" ] && [ -n "$L" ]; then
        bound=$(( (T - 1) * 2 * L ))
        margin=$(( bound - reb ))
    else
        bound="?"; margin="?"
    fi
    printf "%-5s %-7s %-5s %-4s %-9s %-10s %-11s %-9s %-9s %s\n" \
           "$T" "$pass" "${reb:-?}" "${L:-?}" "$bound" "$margin" \
           "${acq:-?}" "${p999:-?}" "${p5:-?}" "${v:-?}"
done
echo
echo "(T-1)KL uses K=2 -- bundle Phase 2 and Phase 4 share one negotiate."
echo "A negative margin is the interesting result: the model does not hold there."
echo "Read the maximum WITH its L; the two come from different counters."
