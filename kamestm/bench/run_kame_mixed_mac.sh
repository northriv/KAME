#!/bin/bash
#
# Mixed-contention sweep on macOS (Apple Silicon: M3 Air, M4 mini, etc.).
# Pair script for ohtaka/run_kame_mixed.sh so the same K × N grid can be
# collected on M-series without SLURM.
#
# Expects binaries at $KAME_ROOT/build/kamestm-tests (same layout as the existing
# bench/build_variants.sh + run_bench.sh). Run once per KAME branch:
#
#   cd ~/KAME && git apply \
#       cpp_test_drafts/kame_payload_integrity_mixed_test.patch
#   cmake --build build -j
#   # master baseline
#   ./run_kame_mixed_mac.sh master > mixed_M4mini_master.tsv
#   # AdaptiveNegotiation
#   ( cd ../kame-adpt && git checkout AdaptiveNegotiation && cmake --build build -j )
#   KAME_ROOT=../kame-adpt ./run_kame_mixed_mac.sh adaptive > mixed_M4mini_adaptive.tsv

set -u

KAME_ROOT=${KAME_ROOT:-$HOME/KAME}
BIN_DIR=${BIN_DIR:-$KAME_ROOT/build/kamestm-tests}
LABEL=${1:-$(hostname -s)}
STRESS=${STRESS:-3}
WARMUP=${WARMUP:-1}
PAYLOAD=${PAYLOAD:-256}
# Cooldown (seconds) between configurations to let thermal/memory state
# recover, so a long N=1→128 × K × test sweep does not penalise the later
# (high-N) cells via accumulated heat / page-cache pressure. Single-config
# runs are unaffected. Override with COOLDOWN=0 to disable.
COOLDOWN=${COOLDOWN:-5}
# Where to drop persisted per-cell stderr logs. Default = same
# directory as this script (which is also where the TSV lands when
# the parent run_mac_variants.sh tee's our stdout).
BENCH_DIR=${BENCH_DIR:-$(cd "$(dirname "$0")" && pwd)}
# Each config runs RUNS independent stress measurements; we print the
# median as the headline rate and the min–max as a [lo-hi] tail so
# bimodal regimes (observed at 2L K=10 N=128 etc.) are visible rather
# than averaged away. Default 3×3s ≈ 9s per config × 8 N × 3 K × 2
# tests ≈ 7 min total, which is short enough to iterate a tuning
# sweep in a coffee break. Override with RUNS=5 for sharper medians.
RUNS=${RUNS:-3}
# M3/M4 Air/mini are 8–10 cores; keep the sweep small enough to finish in
# a few minutes. Override via env for other hosts.
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 16 32 64 128"}
K_LIST=${K_LIST:-"1 2 10 0"}  # K=0 = leaf-only (no bundle); K=1 = all grand (CR=∞); K=2 = CR=2; K=10 = CR=10
TESTS=${TESTS:-"payload_integrity_mixed payload_integrity_3level_mixed"}

# Sanity check.
for T in $TESTS; do
    B="$BIN_DIR/transaction_${T}_test"
    if [ ! -x "$B" ]; then
        echo "ERROR: missing $B" >&2
        exit 1
    fi
done

# --- system-load instrumentation -------------------------------------------
# Records background load so a contaminated cell (e.g. a runaway browser
# stealing cores mid-sweep) can be spotted after the fact instead of silently
# skewing a data point.
loadavg1() {
    if [ -r /proc/loadavg ]; then awk '{print $1}' /proc/loadavg
    elif sysctl -n vm.loadavg >/dev/null 2>&1; then sysctl -n vm.loadavg | awk '{print $2}'
    else uptime | sed -E 's/.*load aver(ages?|age):? *//; s/[, ].*//'; fi
}
system_load_snapshot() {
    echo "  load average (1m): $(loadavg1)"
    echo "  top CPU procs:"
    ps -Ao '%cpu,comm' 2>/dev/null | sort -rn | head -5 | sed 's/^/    /'
}

echo "=== KAME mixed-contention sweep ($LABEL, $(uname -srm)) ==="
echo "Date: $(date)"
echo "CPUs: $(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo ?)"
echo "BIN_DIR: $BIN_DIR"
if git -C "$KAME_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    # NB: use rev-parse, not [ -d .git ] — in a git worktree .git is a *file*.
    echo "KAME rev: $(git -C $KAME_ROOT describe --all --always --dirty 2>/dev/null)${KAME_REV_NOTE:+  [${KAME_REV_NOTE}]}"
    # Fingerprint the transaction source actually being compiled, so a
    # mis-targeted variant overlay (e.g. v0 headers written to the wrong
    # path) is visible in the log instead of silently running master.
    txsrc="$KAME_ROOT/kamestm/transaction.h"
    [ -f "$txsrc" ] || txsrc="$KAME_ROOT/kame/transaction.h"
    if [ -f "$txsrc" ]; then
        echo "tx source: ${txsrc#$KAME_ROOT/}  md5=$( (md5 -q "$txsrc" 2>/dev/null || md5sum "$txsrc" 2>/dev/null | cut -d' ' -f1) )"
    fi
fi
echo "THREADS_LIST='$THREADS_LIST'"
echo "K_LIST='$K_LIST'"
echo "Per-config: warmup=${WARMUP}s stress=${STRESS}s × ${RUNS} runs (median reported) payload=$PAYLOAD cooldown=${COOLDOWN}s"
echo "--- system load at start (watch for non-bench CPU hogs) ---"
system_load_snapshot
echo

# Extract raw counts (leaf_tx, parent/grand_tx, child_updates) and the
# reported commits/s rate in one awk pass. Missing fields → 0 so the
# TSV row stays rectangular.
extract_all() {
    awk '
        {
            if (match($0, /leaf_tx=[0-9]+/))       { s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); leaf=s }
            if (match($0, /(parent|grand)_tx=[0-9]+/)) { s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); pt=s }
            if (match($0, /child_updates=[0-9]+/)) { s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); cu=s }
            if (match($0, /[0-9]+[[:space:]]+commits\/s/)) { s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); rate=s }
        }
        END {
            printf "%d %d %d %d\n", leaf+0, pt+0, cu+0, rate+0
        }
    '
}

# Timeout wrapper — macOS ships gtimeout (brew install coreutils).
TIMEOUT=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout"; fi
if command -v gtimeout >/dev/null 2>&1; then TIMEOUT="gtimeout"; fi

run_one() {
    local bin=$1 sec=$2 thr=$3 pl=$4 k=$5
    local hard=$((sec * 3 + 30))
    # Capture BOTH stdout and stderr to one combined log, so:
    #   (1) extract_all parses the rate line regardless of which fd
    #       the binary used (defends against the regex missing the
    #       `leaf_tx=…` line if a build accidentally routes it to
    #       stderr — exactly the empirical move kitag asked for).
    #   (2) On any anomaly we can persist the EXACT bytes the binary
    #       wrote, so post-mortem grep / diff can answer "did
    #       leaf_tx= even appear?" without having to reproduce.
    local logfile
    logfile=$(mktemp -t kame_mixed.XXXXXX 2>/dev/null || \
              echo "/tmp/kame_mixed.$$.$RANDOM")
    local rc
    if [ -n "$TIMEOUT" ]; then
        "$TIMEOUT" --kill-after=5 "$hard" \
            "$bin" "$sec" "$thr" "$pl" "$k" &>"$logfile"
        rc=$?
    else
        "$bin" "$sec" "$thr" "$pl" "$k" &>"$logfile"
        rc=$?
    fi
    # Force a stdout flush by appending a newline if missing, so the
    # `[…] leaf_tx=…` line (which exits via main()-return → buffered
    # stdout) lands intact in the combined log. macOS / glibc both
    # flush at exit, but if the kernel SIGKILLs the process via the
    # timeout, the buffered stdout never reaches the file. Catching
    # that case is the difference between "extract_all sees leaf_tx="
    # and "extract_all returns 0 0 0 0".
    out=$(extract_all < "$logfile")

    # rc 124/137 = timeout / SIGKILL; rc >= 128 = signalled (SEGV=139).
    # Surface the combined log when something actually went wrong so
    # the happy path stays quiet:
    #   - rc != 0           → always report (crash / timeout / killed)
    #   - rc == 0 && zeros  → report (suspicious — extract_all found no
    #                         leaf_tx= even though the binary exited 0)
    local dump=0
    if [ "$rc" != 0 ]; then
        dump=1
    elif echo "$out" | grep -q '^0 0 0 0'; then
        dump=1
    fi

    local label_safe
    label_safe=$(echo "$LABEL" | tr -c '[:alnum:]_' '_')

    if [ "$dump" = 1 ]; then
        # tee the diagnostic block to BOTH operator stderr AND a per-
        # cell anomaly log so the data survives a `> out.tsv` (which
        # doesn't redirect stderr) and a scrollback wipe.
        local anom="$BENCH_DIR/${label_safe}_K${k}_N${thr}.anom"
        local total
        total=$(wc -l < "$logfile" | tr -d ' ')
        {
            echo "[run_one] bin=$(basename "$bin") K=$k N=$thr sec=$sec rc=$rc"
            # Show the TAIL, not the head: the meaningful output (the
            # `leaf_tx=… commits/s` result line, an assertion failure, or a
            # crash backtrace) lands at the END of the log, whereas the head
            # is flooded by `Reserve swap space …` pool-init lines. When a
            # cell reports a suspicious 0 we want to see how the run actually
            # finished, which is the last lines.
            echo "[run_one] combined stdout+stderr ($total lines; last 30):"
            tail -30 "$logfile"
            if [ "$total" -gt 30 ]; then
                echo "[run_one] (... see ${label_safe}_K${k}_N${thr}.log for full content)"
            fi
        } | tee -a "$anom" >&2
    fi

    # Probe-event accounting. grep -c warnings on empty file → /dev/null.
    local n_ll=0 n_mode=0
    if [ -s "$logfile" ]; then
        n_ll=$(grep -c 'verdict=LIVELOCK' "$logfile" 2>/dev/null || echo 0)
        n_mode=$(grep -c '\[ll-probe\] mode:' "$logfile" 2>/dev/null || echo 0)
        # Always persist the combined log when non-empty — kitag's
        # directive: keep the raw bytes around so we can post-mortem
        # any cell, not just the ones the dump heuristic flagged.
        # Append with a per-run banner so `tail -100 …K10_N8.log`
        # extracts the latest run cleanly even when RUNS independent
        # stress measurements share one log per K,N cell.
        {
            echo "==== run start: $(date +%T)" \
                 "bin=$(basename "$bin")" \
                 "K=$k N=$thr sec=$sec rc=$rc" \
                 "leaf_tx_grep=$(grep -c 'leaf_tx=' "$logfile") ===="
            cat "$logfile"
            echo "==== run end:   $(date +%T) ===="
        } >> "$BENCH_DIR/${label_safe}_K${k}_N${thr}.log"
    fi
    rm -f "$logfile"
    echo "$out $n_ll $n_mode"
}

# Pick median / min / max from a list of sorted integer rates.
stats_med_min_max() {
    awk 'BEGIN{n=0}
         { v[n++]=$1 }
         END{
             # selection sort (n small, RUNS ≤ ~10)
             for(i=0;i<n-1;i++)
                 for(j=i+1;j<n;j++)
                     if(v[i]>v[j]){t=v[i];v[i]=v[j];v[j]=t}
             med = (n%2==1) ? v[(n-1)/2] : int((v[n/2-1]+v[n/2])/2)
             printf "%d %d %d", med, v[0], v[n-1]
         }'
}

printf "%-32s\t%-3s\t%-5s\t%-12s\t%-12s\t%-14s\t%-12s\t%-12s\t%-6s\t%-6s\t%-6s\n" \
       test K N leaf_med "parent/grand_med" child_med child_upd/s "[min-max]" LL_med MODE_med load1
for T in $TESTS; do
    B="$BIN_DIR/transaction_${T}_test"
    for K in $K_LIST; do
        for N in $THREADS_LIST; do
            # Warmup once, discard output; amortises initial pool/mmap
            # setup across the RUNS measurements below.
            if [ -n "$TIMEOUT" ]; then
                $TIMEOUT --kill-after=5 $((WARMUP * 3 + 30)) \
                    "$B" "$WARMUP" "$N" "$PAYLOAD" "$K" > /dev/null 2>&1
            else
                "$B" "$WARMUP" "$N" "$PAYLOAD" "$K" > /dev/null 2>&1
            fi
            # Collect RUNS independent stress runs, remember each column
            # so we can median-aggregate all of them (not just the rate).
            # ll_list / mode_list track per-run probe-event counts harvested
            # by run_one from the binary's stderr stream — appearing as the
            # 5th and 6th fields of its echo line.
            leaf_list=""; pt_list=""; cu_list=""; rate_list=""
            ll_list="";  mode_list=""
            for r in $(seq 1 "$RUNS"); do
                line=$(run_one "$B" "$STRESS" "$N" "$PAYLOAD" "$K")
                [ -z "$line" ] && line="0 0 0 0 0 0"
                # shellcheck disable=SC2086
                set -- $line
                leaf_list="$leaf_list
$1"
                pt_list="$pt_list
$2"
                cu_list="$cu_list
$3"
                rate_list="$rate_list
$4"
                ll_list="$ll_list
${5:-0}"
                mode_list="$mode_list
${6:-0}"
            done
            # Drop the empty first line from the heredoc-style builds.
            read leaf leaf_min leaf_max < <(printf "%s" "$leaf_list" | sed '/^$/d' | stats_med_min_max)
            read pt   pt_min   pt_max   < <(printf "%s" "$pt_list"   | sed '/^$/d' | stats_med_min_max)
            read cu   cu_min   cu_max   < <(printf "%s" "$cu_list"   | sed '/^$/d' | stats_med_min_max)
            read rate rate_min rate_max < <(printf "%s" "$rate_list" | sed '/^$/d' | stats_med_min_max)
            read ll   ll_min   ll_max   < <(printf "%s" "$ll_list"   | sed '/^$/d' | stats_med_min_max)
            read mode mode_min mode_max < <(printf "%s" "$mode_list" | sed '/^$/d' | stats_med_min_max)
            # 1-min load sampled at this cell: if load1 >> N the cell shared
            # cores with something else (e.g. a browser) and is suspect.
            cfg_load=$(loadavg1)
            printf "%-32s\t%-3d\t%-5d\t%-12s\t%-12s\t%-14s\t%-12d\t[%d-%d]\t%-6d\t%-6d\t%-6s\n" \
                   "$T" "$K" "$N" "$leaf" "$pt" "$cu" "$rate" "$rate_min" "$rate_max" "$ll" "$mode" "$cfg_load"
            # Cooldown between configurations: let thermal/page-cache state
            # recover so later (high-N) cells are not penalised by the heat
            # and memory pressure accumulated earlier in the sweep.
            [ "$COOLDOWN" -gt 0 ] && sleep "$COOLDOWN"
        done
    done
done
echo
echo "--- system load at end ---"
system_load_snapshot
echo "End: $(date)"
