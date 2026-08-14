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
# rule was -- survives only the single-peer case.  Sweeping T is what tests it.
#
# READ THE CPU BUDGET BEFORE READING ANY RESULT.  The bound's T counts peers
# that can be IN FLIGHT AT ONCE.  Raising T past the number of CPUs the run is
# allowed does not raise that; it raises preemption, and the measured maximum
# stops responding to T entirely -- it is then rate-limited rather than
# interleaving-limited, and the sweep measures the scheduler.  The first run of
# this script (2026-08-14) went out over CPUS="2,3" and produced exactly that:
# 2, 2, 14, 5, 6, 9, 9, 6, 5, 5 for T = 3..12, no trend, T=12 among the lowest,
# every margin comfortably positive and none of it meaning anything.  So:
#
#     A T-SWEEP NEEDS len(CPUS) >= max(TLIST).
#
# Set CPUS to a list that large, or do not sweep.  The script refuses by
# default rather than produce another table of scheduler measurements.
#
# THE WRAPPERS ARE NOT OPTIONAL.  This is the house recipe, and every element
# of it was learned by losing a measurement without it:
#
#   with_pmqos   holds /dev/cpu_dma_latency at 0.  Without it the tail carries
#                C-state exit latency and no absolute number is quotable.  Its
#                own header carries the measured floor table.  It is staged
#                next to the binaries by the `with_pmqos` CMake target, and
#                reaching across trees for it has twice produced a silent
#                "command not found" -- silent because the usual invocation
#                pipes stderr into a grep.  This script checks for it and stops.
#   taskset      confines the run to dedicated CPUs.  Note that `nproc` honours
#                the affinity mask, so inside taskset it reports the SIZE OF
#                THE MASK, not the machine.  Do not read it as a core count.
#   OS_FIFO=1    puts acquisition on SCHED_FIFO.  Without it HIGHEST is a
#                library priority only and the run is not a realtime run.
#   OS_PIN=1     acquisition alone on the last CPU of the mask, everyone else
#                on the first -- the isolcpus shape KAME deploys.  Peers then
#                share one CPU, which is right for the deployment measurement
#                and wrong for a T sweep; PIN defaults to 1 and the sweep guard
#                above is what keeps the two from being confused.
#   SLOW_NS      the slow-commit threshold.  Set it near p99.9 so `slow_n` is a
#                usable COUNT.  Left at the default it selects a population of
#                zero to six per run and every statistic conditioned on it is
#                noise -- which is how a 4.60 -> 0.81 "improvement" got
#                published on this branch and then had to be withdrawn.
#
# WHAT TO READ.  slow_n (a count) and the rebuild maximum WITH its L.  Not the
# averages conditioned on slow commits, and not MAX alone -- MAX is dominated
# by the wait budget and by the scheduler, and swung 51.7 to 167.6 ms between
# two runs of the same binary here.
#
# AND: n=2 does not resolve a trend.  T=4 gave 14 and 5 in the same pair.  A
# single 30-minute soak on this branch read "over by one in 115 M commits" and
# four repeats put the excursion at +9 to +26.  REPS=3 minimum for any claim.
#
# Usage:
#   tests/rt_sweep_threads.sh <path-to-transaction_priority_mixed_test> [outdir]
# Environment:
#   CPUS="2,3"   taskset list.  A T sweep needs at least max(TLIST) of them.
#   TLIST="4"    thread counts.  T = 1 (acq) + 1 (UI) + scripting + normals.
#   SECS=900     seconds per run
#   REPS=3       ascending+descending passes
#   PIN=1        KAME_MIX_OS_PIN
#   FIFO=1       KAME_MIX_OS_FIFO
#   SLOW_NS=7000 KAME_MIX_SLOW_NS
#   PMQOS=1      0 skips with_pmqos (then no absolute number is quotable)
#   FORCE=0      1 sweeps T past the CPU budget anyway, for scheduler studies

set -u

BIN=${1:?usage: rt_sweep_threads.sh <binary> [outdir]}
OUT=${2:-rt_sweep_$$}
CPUS=${CPUS:-2,3}
TLIST=${TLIST:-4}
SECS=${SECS:-900}
REPS=${REPS:-3}
PIN=${PIN:-1}
FIFO=${FIFO:-1}
SLOW_NS=${SLOW_NS:-7000}
PMQOS=${PMQOS:-1}
FORCE=${FORCE:-0}

[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 1; }
BINDIR=$(cd "$(dirname "$BIN")" && pwd)

# with_pmqos is staged next to the binaries; the build dir may nest one level.
PMQOS_BIN=""
if [ "$PMQOS" = 1 ]; then
    for c in "$BINDIR/with_pmqos" "$BINDIR/../with_pmqos"; do
        [ -x "$c" ] && { PMQOS_BIN=$c; break; }
    done
    [ -n "$PMQOS_BIN" ] || {
        echo "with_pmqos not found next to $BINDIR -- build the with_pmqos" >&2
        echo "target, or set PMQOS=0 and stop quoting absolute latencies." >&2
        exit 1; }
fi

ncpus=$(echo "$CPUS" | tr ',' '\n' | while read -r r; do
            case $r in *-*) seq "${r%-*}" "${r#*-}";; *) echo "$r";; esac
        done | wc -l)
maxT=$(echo "$TLIST" | tr ' ' '\n' | sort -n | tail -1)
if [ "$maxT" -gt "$ncpus" ] && [ "$FORCE" != 1 ]; then
    cat >&2 <<MSG
REFUSING: TLIST reaches T=$maxT but CPUS='$CPUS' gives only $ncpus CPU(s).

The bound's T counts peers in flight AT ONCE.  Past the CPU budget T adds
preemption, not concurrency, and the measured maximum stops responding to it --
the sweep then measures the scheduler.  That has already been run once and the
table looked fine while meaning nothing.

Either widen CPUS to >= $maxT entries, or narrow TLIST, or set FORCE=1 if the
scheduler is what you are actually studying.
MSG
    exit 2
fi

mkdir -p "$OUT" || exit 1

{
    echo "date          $(date -Is)"
    echo "host          $(uname -n)"
    echo "kernel        $(uname -a)"
    echo "cpus (mask)   $CPUS  -> $ncpus CPU(s)"
    echo "nproc (masked)$(taskset -c "$CPUS" nproc 2>/dev/null)  <- mask size, NOT the machine"
    echo "nproc (all)   $(nproc --all 2>/dev/null)"
    echo "clocksource   $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null)"
    echo "governor      $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "with_pmqos    ${PMQOS_BIN:-<skipped>}"
    echo "binary        $BIN"
    echo "binary sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "git rev       $(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null)"
    echo "git dirty     $(git -C "$(dirname "$0")" status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "SECS=$SECS TLIST='$TLIST' REPS=$REPS PIN=$PIN FIFO=$FIFO SLOW_NS=$SLOW_NS"
} | tee "$OUT/provenance.txt"
echo

run_one() {   # $1 = T, $2 = tag
    local T=$1 tag=$2 rest scr nrm log
    rest=$((T - 2)); [ "$rest" -lt 1 ] && rest=1
    scr=$((rest / 2)); nrm=$((rest - scr))
    log="$OUT/T${T}.${tag}.log"
    ${PMQOS_BIN:+$PMQOS_BIN} taskset -c "$CPUS" env \
        KAME_MIX_SECS=$SECS \
        KAME_MIX_SLOW_NS=$SLOW_NS \
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

echo
printf "%-4s %-7s %-6s %-5s %-4s %-9s %-9s %-10s %-8s %s\n" \
       T pass slow_n reb L "(T-1)KL" "acq/s" p99.9 MAX verdict
for f in "$OUT"/T*.log; do
    T=$(basename "$f" | sed 's/^T\([0-9]*\)\..*/\1/')
    pass=$(basename "$f" | sed 's/^T[0-9]*\.\(.*\)\.log/\1/')
    # slow_n: the COUNT of commits past SLOW_NS, not an average over them.
    sn=$(grep -a "slow(>=" "$f" | grep -oE "n=[0-9]+" | head -1 | cut -d= -f2)
    reb=$(grep -a "rebuilds: max=" "$f" | grep -oE "max=[0-9]+" | head -1 | cut -d= -f2)
    L=$(grep -a "rebuilds: max=" "$f" | grep -oE "L=[0-9]+" | head -1 | cut -d= -f2)
    acq=$(grep -a "acq(record)" "$f" | grep -oE "\([0-9]+ /s" | tr -d '( /s')
    p999=$(grep -a "^    n=" "$f" | grep -oE "p99\.9=[0-9]+" | head -1 | cut -d= -f2)
    mx=$(grep -a "^    n=" "$f" | grep -oE "MAX=[0-9]+" | head -1 | cut -d= -f2)
    v=$(grep -aoE "^PASSED|^FAILED|STALL" "$f" | head -1)
    if [ -n "$reb" ] && [ -n "$L" ]; then bound=$(( (T-1) * 2 * L )); else bound="?"; fi
    printf "%-4s %-7s %-6s %-5s %-4s %-9s %-9s %-10s %-8s %s\n" \
           "$T" "$pass" "${sn:-?}" "${reb:-?}" "${L:-?}" "$bound" \
           "${acq:-?}" "${p999:-?}" "${mx:-?}" "${v:-?}"
done
echo
echo "slow_n is the count past SLOW_NS=$SLOW_NS -- the tail statistic to read."
echo "(T-1)KL uses K=2; a negative margin is the informative result."
echo "MAX is dominated by the wait budget and the scheduler; do not read it alone."
