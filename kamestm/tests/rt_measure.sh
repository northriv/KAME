#!/bin/bash
# Copyright (C) 2002-2026 Kentaro Kitagawa
#
# Host floor characterisation and absolute-latency measurement, in one place
# so that neither the procedure nor the reason for each step has to be
# remembered.  Everything here was learned by getting it wrong first; the
# comments say which, because a step whose reason is forgotten is a step that
# gets dropped.
#
#   rt_measure.sh floor   [outdir]
#   rt_measure.sh latency <binary> [outdir]        ROW=1|2|3
#
# ROW picks the README realtime-table row to reproduce, with that row's knobs
# and duration: 1 = contended (default, 300 s), 2 = uncontended
# (KAME_MIX_DISJOINT=1, 60 s), 3 = budgeted NORMAL (KAME_MIX_ACQ_NORMAL=1,
# 120 s).  SECS overrides the duration and says so in the provenance.
#
# Run it as `sudo CPUS=2,3 rt_measure.sh ...` and NOT `sudo -E` -- sudo ignores
# -E ("preserving the entire environment is not supported") and the knobs then
# silently fall back to their defaults.
#
# WHY FLOOR FIRST.  kamestm/README.md opens its realtime table with "measure
# the host's floor before quoting any number here", and gives this host's own
# range: 67.9 us (no isolation) -> 17.0 us (isolcpus/nohz_full) -> 219 ns
# (+ PM-QoS).  A latency figure read against the wrong floor attributes the
# machine to the STM.  On 2026-08-14 a 39.4 us MAX was reported here as an STM
# number while `rtla osnoise` was showing the machine alone taking 63 us out of
# a spinning thread -- the measurement was under its own floor and said nothing.
#
# WHAT THE TWO TRACERS ANSWER.  They are not interchangeable:
#   hwnoise  runs with interrupts DISABLED, so nothing software can intrude.
#            What it reports is firmware: SMI and friends.  Zero here means the
#            machine has no hardware floor and absolute numbers are meaningful.
#            Non-zero is largely unfixable from the OS -- BIOS (USB legacy
#            support is the classic source), or a different machine.
#   osnoise  runs with interrupts ENABLED.  The difference between the two is
#            everything software: IRQ entry/exit, C-state exits, P-state
#            transitions.  That difference is what PM-QoS and the governor act
#            on, which is why this script measures osnoise twice -- once bare,
#            once with both applied -- rather than assuming the fix works.
#
# THE BUILD DECIDES WHAT MAY BE QUOTED.  KAME_STM_NEG_DIAG puts two clock reads
# per bundle INSIDE the measured path (README says so explicitly), so a diag
# build measures a different program.  `latency` detects a diag binary from its
# own output and refuses.  For the rebuild bound, which needs those counters,
# use tests/rt_sweep_threads.sh and do not quote its latency columns.

set -u

MODE=${1:?usage: rt_measure.sh floor [outdir] | rt_measure.sh latency <binary> [outdir]}
shift

CPUS=${CPUS:-2,3}
# ROW selects which README realtime-table row is being reproduced.  The knobs
# and the duration are part of the row's definition, not of the operator's
# memory: row 2 is the uncontended path and row 3 is the budgeted NORMAL tier,
# and getting either wrong reproduces a different row while looking right.
ROW=${ROW:-1}
SECS_SET=${SECS+set}
case "$ROW" in
    1) ROW_ENV="";                      ROW_SECS=300
       ROW_DESC="HIGHEST, 5-node commit, peers writing into the same subtree" ;;
    2) ROW_ENV="KAME_MIX_DISJOINT=1";   ROW_SECS=60
       ROW_DESC="the same commit with no peer on its subtree" ;;
    3) ROW_ENV="KAME_MIX_ACQ_NORMAL=1"; ROW_SECS=120
       ROW_DESC="NORMAL under the 20 ms budget" ;;
    *) echo "ROW must be 1, 2 or 3 (README's realtime table)" >&2; exit 1 ;;
esac
SECS=${SECS:-$ROW_SECS}
REPS=${REPS:-3}
SLOW_NS=${SLOW_NS:-7000}
HWNOISE_SECS=${HWNOISE_SECS:-300}
OSNOISE_SECS=${OSNOISE_SECS:-60}

TRACING=/sys/kernel/tracing
[ -d "$TRACING" ] || TRACING=/sys/kernel/debug/tracing

say() { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------"; }

need_root() {
    [ "$(id -u)" = 0 ] || {
        say "This mode needs root (rtla, the tracer, /dev/cpu_dma_latency)."
        say "Re-run under sudo -E."
        exit 1; }
}

# The redirection itself fails noisily when the tracefs is absent, so guard on
# the file rather than on the exit status.
reset_tracer() {
    [ -w "$TRACING/current_tracer" ] && echo nop > "$TRACING/current_tracer"
    return 0
}

# Governor is global state; put it back even on Ctrl-C.
GOV_SAVED=""
save_governors() {
    GOV_SAVED=""
    for c in $(echo "$CPUS" | tr ',' ' '); do
        local f=/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor
        [ -r "$f" ] && GOV_SAVED="$GOV_SAVED $c:$(cat $f)"
    done
}
restore_governors() {
    for e in $GOV_SAVED; do
        echo "${e#*:}" > /sys/devices/system/cpu/cpu${e%%:*}/cpufreq/scaling_governor 2>/dev/null || true
    done
}
set_performance() {
    for c in $(echo "$CPUS" | tr ',' ' '); do
        echo performance > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true
    done
}
trap 'restore_governors; reset_tracer' EXIT INT TERM

provenance() {
    say "date          $(date -Is)"
    say "host          $(uname -n)"
    say "kernel        $(uname -r)"
    # The CPU model and its core/thread split are part of the result, not
    # colour: an SMT sibling sharing the isolated core's execution units, or a
    # different microarchitecture, moves these numbers more than most of the
    # knobs below.  A reader cannot re-derive either from the hostname.
    say "cpu model     $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)"
    local _cores; _cores=$(sed -n 's/^core id[[:space:]]*: //p' /proc/cpuinfo | sort -u | wc -l)
    say "cpu topology  $(nproc --all) thread(s) over ${_cores:-?} core id(s)$([ "$(nproc --all)" != "$_cores" ] && echo '  <- SMT: check the isolated pair are not siblings')"
    say "cmdline       $(cat /proc/cmdline)"
    # Each of these moves the floor by roughly an order of magnitude (README's
    # 67.9 us / 17.0 us / 219 ns table), so their presence is part of the result.
    for k in isolcpus nohz_full rcu_nocbs irqaffinity nmi_watchdog; do
        if grep -q -- "$k=" /proc/cmdline; then
            say "  $k$(printf '%*s' $((14 - ${#k})) '')PRESENT"
        else
            say "  $k$(printf '%*s' $((14 - ${#k})) '')ABSENT  <- the floor will be higher"
        fi
    done
    say "clocksource   $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null)"
    for c in $(echo "$CPUS" | tr ',' ' '); do
        say "governor cpu$c $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null)"
    done
    # RT throttling: 950000/1000000 by default, i.e. a 50 ms hole once a second
    # for anything that stays runnable at SCHED_FIFO.  A spinning RT thread
    # WILL hit it.  Not a bug, but it must not be mistaken for the STM.
    say "rt_runtime_us $(cat /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null)  (950000 = 50 ms/s handed to non-RT)"
    say "rt_period_us  $(cat /proc/sys/kernel/sched_rt_period_us 2>/dev/null)"
    say "rtprio limit  $(ulimit -r 2>/dev/null)"
    say "cpu_dma_lat   $([ -w /dev/cpu_dma_latency ] && echo writable || echo 'NOT writable  <- PM-QoS will be refused')"
    say "cpus (mask)   $CPUS"
    say "nproc (all)   $(nproc --all)"
}

# rtla osnoise/hwnoise print a header then one row per CPU; pull Max Single.
max_single() { awk '$1 ~ /^[0-9]+$/ {print $7}' "$1" | sort -n | tail -1; }
pct_aval()   { awk '$1 ~ /^[0-9]+$/ {print $5}' "$1" | sort -n | head -1; }

do_floor() {
    local OUT=${1:-rt_floor_$(date +%Y%m%d_%H%M%S)}
    need_root
    command -v rtla >/dev/null || { say "rtla not found (package: linux-tools / rtla)"; exit 1; }
    mkdir -p "$OUT" || exit 1

    provenance | tee "$OUT/provenance.txt"
    rule

    # Resolve with_pmqos once; steps 1 and 3 both need it.
    PMQOS_CMD=""
    for c in ./with_pmqos "$(command -v with_pmqos 2>/dev/null)"; do
        [ -n "$c" ] && [ -x "$c" ] && { PMQOS_CMD=$c; break; }
    done
    [ -n "$PMQOS_CMD" ] || say "NOTE: with_pmqos not found in cwd or PATH -- steps 1 and 3 lose their C-state control."

    # 1. FIRMWARE.  Interrupts disabled, so nothing SOFTWARE can intrude and
    #    what remains is SMI-class -- the question that decides whether the host
    #    can carry an absolute number at all.
    #
    #    UNDER PM-QoS, and that is not optional: "interrupts disabled" does not
    #    mean "no C-states".  The tracer samples 750 ms of each second and the
    #    CPU is free in between, and the PACKAGE can drop deep while the
    #    housekeeping CPUs idle, which slows the measured core through the
    #    uncore and the shared cache.  Both land in the unattributed bucket and
    #    read as firmware.  Run bare on this host it reported Max Single 57 us
    #    and 16323 events -- indistinguishable from the bare osnoise figure of
    #    60 us, which is the tell -- while a hand run at another moment reported
    #    exactly zero.  Two runs of the same length disagreeing that way is not
    #    a firmware measurement; it is a C-state measurement.
    say "[1/3] hwnoise ${HWNOISE_SECS}s -- firmware (SMI) floor, IRQs off, under PM-QoS"
    reset_tracer
    ${PMQOS_CMD:+$PMQOS_CMD} rtla hwnoise top -c "$CPUS" -d "${HWNOISE_SECS}s" > "$OUT/hwnoise.txt" 2>&1
    tail -4 "$OUT/hwnoise.txt"
    local hw_max; hw_max=$(max_single "$OUT/hwnoise.txt")
    rule

    # 2. BARE.  Interrupts enabled, no PM-QoS, governor as found.  This is the
    #    floor a run gets when nobody applied the recipe -- which is what
    #    happened on 2026-08-14.
    say "[2/3] osnoise ${OSNOISE_SECS}s -- bare (no PM-QoS, governor as found)"
    reset_tracer
    rtla osnoise top -c "$CPUS" -d "${OSNOISE_SECS}s" > "$OUT/osnoise_bare.txt" 2>&1
    tail -4 "$OUT/osnoise_bare.txt"
    local bare_max; bare_max=$(max_single "$OUT/osnoise_bare.txt")
    rule

    # 3. TREATED.  PM-QoS held and the governor pinned to performance.  The
    #    delta against (2) is exactly what those two buy, measured rather than
    #    assumed -- C-state exits and P-state transitions are invisible to
    #    hwnoise (a spinning IRQ-off thread never idles) and both land in
    #    osnoise's unattributed bucket, where they look like firmware.
    say "[3/3] osnoise ${OSNOISE_SECS}s -- with PM-QoS + performance governor"
    save_governors; set_performance
    reset_tracer
    ${PMQOS_CMD:+$PMQOS_CMD} rtla osnoise top -c "$CPUS" -d "${OSNOISE_SECS}s" > "$OUT/osnoise_treated.txt" 2>&1
    tail -4 "$OUT/osnoise_treated.txt"
    local treated_max; treated_max=$(max_single "$OUT/osnoise_treated.txt")
    restore_governors
    rule

    {
        say "FLOOR SUMMARY  (Max Single, us -- the largest single hole in a spinning thread)"
        say "  firmware (hwnoise, IRQ off)      ${hw_max:-?}"
        say "  bare     (osnoise)               ${bare_max:-?}"
        say "  treated  (osnoise + PMQoS + perf) ${treated_max:-?}"
        say ""
        if [ "${hw_max:-1}" = 0 ]; then
            say "  Firmware floor is ZERO: no SMI.  Absolute latency numbers are"
            say "  meaningful on this host, and whatever osnoise sees is software."
        elif [ -n "${bare_max:-}" ] && [ "${hw_max:-0}" -ge $(( bare_max * 8 / 10 )) ]; then
            say "  Firmware reads ${hw_max} us, but that is within 20% of the BARE"
            say "  osnoise figure (${bare_max} us) -- the two are measuring the same"
            say "  thing, and it is not firmware.  Suspect residual C-state or"
            say "  package-idle effects reaching the core, and treat the TREATED"
            say "  number as the floor."
        else
            say "  Firmware floor is ${hw_max} us and is distinct from the software"
            say "  figures.  No STM number below that is attributable to the STM."
            say "  BIOS (USB legacy support first) or a different machine."
        fi
        say ""
        say "  QUOTE NOTHING BELOW THE TREATED FLOOR.  Run 'latency' under the same"
        say "  PM-QoS and governor this step used, and compare against ${treated_max:-?} us."
    } | tee "$OUT/FLOOR.txt"
    say ""
    say "wrote $OUT/"
}

do_latency() {
    local BIN=${1:?usage: rt_measure.sh latency <binary> [outdir]}
    local OUT=${2:-rt_latency_$(date +%Y%m%d_%H%M%S)}
    [ -x "$BIN" ] || { say "not executable: $BIN"; exit 1; }
    local BINDIR; BINDIR=$(cd "$(dirname "$BIN")" && pwd)

    # PRE-FLIGHT.  Both of these degrade to a warning INSIDE the run and are
    # then invisible in any summary that does not look for them.  The
    # 2026-08-14 set went out with neither and was compared against README
    # numbers taken with both.
    local fail=0
    [ -w /dev/cpu_dma_latency ] || {
        say "PRE-FLIGHT: /dev/cpu_dma_latency not writable -- with_pmqos will warn"
        say "  and continue, and C-state exits will be IN the numbers."; fail=1; }
    if [ "$(ulimit -r 2>/dev/null || echo 0)" = 0 ] && [ "$(id -u)" != 0 ]; then
        say "PRE-FLIGHT: RLIMIT_RTPRIO is 0 and not root -- SCHED_FIFO will be"
        say "  REFUSED and the run silently becomes SCHED_OTHER throughout."; fail=1
    fi
    [ "$fail" = 0 ] || { say "Run under sudo -E, or set IGNORE_PREFLIGHT=1 deliberately."
        [ "${IGNORE_PREFLIGHT:-0}" = 1 ] || exit 3; }

    local pm="$BINDIR/with_pmqos"
    [ -x "$pm" ] || pm="$BINDIR/../with_pmqos"
    [ -x "$pm" ] || { say "with_pmqos not found next to the binary; build the target."; exit 1; }

    mkdir -p "$OUT" || exit 1
    { provenance
      say "binary        $BIN"
      say "binary sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
      say "git rev       $(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null)"
      say "row           $ROW -- $ROW_DESC"
      say "row env       ${ROW_ENV:-<none>}"
      say "SECS=$SECS${SECS_SET:+ (overridden; this row default is $ROW_SECS)} REPS=$REPS SLOW_NS=$SLOW_NS"
    } | tee "$OUT/provenance.txt"
    say ""

    save_governors; set_performance

    # Record the composed command.  KAME_MIX_DISJOINT and KAME_MIX_ACQ_NORMAL
    # change which row is being reproduced and neither appears in the run's own
    # banner, so "did the knob reach the binary" is otherwise unanswerable from
    # the artefacts.
    say "invocation    $pm taskset -c $CPUS env KAME_MIX_SECS=$SECS KAME_MIX_SLOW_NS=$SLOW_NS KAME_MIX_OS_FIFO=1 KAME_MIX_OS_PIN=1 ${ROW_ENV:-} $BIN" \
        | tee -a "$OUT/provenance.txt"
    say ""

    for r in $(seq 1 "$REPS"); do
        "$pm" taskset -c "$CPUS" env \
            KAME_MIX_SECS=$SECS KAME_MIX_SLOW_NS=$SLOW_NS \
            KAME_MIX_OS_FIFO=1 KAME_MIX_OS_PIN=1 \
            ${ROW_ENV:+$ROW_ENV} \
            "$BIN" > "$OUT/run$r.log" 2>&1
        say "run$r exit=$? -> $OUT/run$r.log"
    done
    restore_governors

    # A diag build measures a different program: two clock reads per bundle
    # inside the path.  Detect it from its own output rather than trusting the
    # caller to remember which build tree this was.
    if grep -qa "livelock probe over ALL" "$OUT"/run1.log; then
        say ""
        say "*** THIS IS A KAME_STM_NEG_DIAG BUILD.  Its pass timer sits inside the"
        say "*** measured path.  These latencies are NOT comparable with README's"
        say "*** table or with any plain-Release number.  Rebuild without the define."
    fi

    say ""
    # `n` is the warm commit count, and it is not decoration: latency_hist.h
    # prints a percentile only when at least 10 samples fall beyond it
    # (n*(1-p) >= 10), so n is what says whether the p99.999 column means
    # anything -- a quoted p99.999 needs n >= 1 M.  A reader of the README
    # table should not have to infer the sample size from the run length.
    printf "%-6s %-12s %-8s %-8s %-9s %-10s %-10s %-8s %s\n" \
           run n p50 p99 p99.9 p99.999 MAX slow_n verdict
    for f in "$OUT"/run*.log; do
        local line; line=$(grep -a "^    n=" "$f" | head -1)
        local n p50 p99 p999 p5 mx sn v
        n=$(echo "$line" | grep -oE "^    n=[0-9]+" | cut -d= -f2)
        p50=$(echo "$line" | grep -oE "p50=[0-9]+" | cut -d= -f2)
        p99=$(echo "$line" | grep -oE "p99=[0-9]+" | cut -d= -f2)
        p999=$(echo "$line" | grep -oE "p99\.9=[0-9]+" | cut -d= -f2)
        p5=$(echo "$line" | grep -oE "p99\.999=[0-9]+" | cut -d= -f2)
        mx=$(echo "$line" | grep -oE "MAX=[0-9]+" | cut -d= -f2)
        sn=$(grep -a "slow(>=" "$f" | grep -oE "n=[0-9]+" | head -1 | cut -d= -f2)
        v=$(grep -aoE "^PASSED|^FAILED|STALL" "$f" | head -1)
        grep -qa "could not hold /dev/cpu_dma_latency" "$f" && v="$v NO-PMQOS"
        grep -qa "SCHED_FIFO .* not permitted" "$f"        && v="$v NO-FIFO"
        printf "%-6s %-12s %-8s %-8s %-9s %-10s %-10s %-8s %s\n" \
               "$(basename "$f" .log)" "${n:-?}" "${p50:-?}" "${p99:-?}" "${p999:-?}" \
               "${p5:-?}" "${mx:-?}" "${sn:-?}" "${v:-?}"
    done
    say ""
    say "n = warm commits; a '?' in p99.999 means n was too small to support it."
    say "Latencies in ns.  README quotes the WORST of the runs for its table row."
    say "Compare MAX against the treated floor from 'rt_measure.sh floor' --"
    say "a MAX under the floor is the machine, not the STM."
}

case "$MODE" in
    floor)   do_floor   "$@" ;;
    latency) do_latency "$@" ;;
    *) say "usage: rt_measure.sh floor [outdir] | rt_measure.sh latency <binary> [outdir]"; exit 1 ;;
esac
