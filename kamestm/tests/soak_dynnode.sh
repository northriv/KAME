#!/bin/sh
# Long soak for the transaction_dynamic_node_test failure seen once on i486
# (2026-08-22).  The point is NOT throughput: it is that when the failure
# recurs, the evidence survives.  The original occurrence taught us almost
# nothing, because ctest overwrote Testing/Temporary/LastTest.log on the very
# next run — we knew only WHICH test failed, not which assertion fired.
#
# It reproduced on the first round once this harness existed, and the kept
# output settled the question: the run ends at a bare `failed1` with no
# `Gn1:`..`Gn4:` lines after it.  Two sites print `failed1`; only the payload
# one prints the Gn values, so this is the OTHER one — `objcnt != 0` after
# every node has been reset.  A live-object leak, not payload corruption.
# `objcnt` is `atomic<int>`, so it is not a counter race either.
#
# What the failure is NOT, established while chasing it:
#   * Not a timeout.  No TIMEOUT property is set (ctest allows 1500 s) and the
#     test runs in ~20 s.
#   * Not the HANG watchdog.  KAME_STM_HANG_ABORT_N is 0 under NDEBUG and the
#     tree is Release, so a livelock there hangs rather than aborting.
#   * Not stamp wraparound.  Shrinking STAMP_US_BITS from 24 to 8 — a 0.128 ms
#     half-window, ~65000x tighter than compact mode's real 8.3 s — still
#     passes: every comparison here is between contenders microseconds apart.
#   * Not a data race TSan can see.  A full run under TSan (compact mode, pool
#     allocator both on and off) reports races on exactly one variable,
#     `Linkage::m_tx_commit_count`, a livelock-probe counter; allocator.cpp
#     never appears.  Worth fixing on its own (its "single writer" comment is
#     disproved by the write-write races) but it cannot leak an object.
#
# Two arms, both i486, because the strict build changes timing and may not
# reproduce what Release does:
#   rel     the exact Release build that failed.
#   strict  -UNDEBUG -DTRANSACTIONAL_STRICT_assert -O1: 30 asserts plus 12
#           STRICT_assert packet-consistency checks live, and HANG_ABORT_N back
#           to 3 so a livelock aborts with a dump.  Localises a fault where it
#           happens rather than at the end-of-test tally.
#
# Note for whoever configures the strict tree: CMake appends
# CMAKE_CXX_FLAGS_<CONFIG> AFTER CMAKE_CXX_FLAGS, so `-UNDEBUG` in the latter
# is undone by RelWithDebInfo's own `-DNDEBUG` (and `-O1` by its `-O2`).  Clear
# CMAKE_CXX_FLAGS_RELWITHDEBINFO, then confirm the object really references
# __assert_fail before trusting the arm.
#
# Usage:  soak_dynnode.sh [rounds] [parallel-per-arm] [outdir]
# Detach: setsid nohup soak_dynnode.sh 400 2 /tmp/soak >/dev/null 2>&1 </dev/null &
#   setsid is not decoration — a plain background job was reaped with its
#   parent shell twice during this investigation.

ROUNDS=${1:-200}
PAR=${2:-2}
OUT=${3:-/tmp/soak_dynnode}

REL=${REL:-/tmp/bt-486/kamestm-tests/transaction_dynamic_node_test}
STRICT=${STRICT:-/tmp/bt-486s/kamestm-tests/transaction_dynamic_node_test}

# Per-run cap.  The test takes ~20 s; 300 s means wedged, and a wedged run must
# be recorded rather than allowed to stall the soak.  A hang is itself a result:
# the rel arm cannot abort on livelock (HANG_ABORT_N == 0 under NDEBUG), so it
# would simply never return.
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-300}

for b in "$REL" "$STRICT"; do
    [ -x "$b" ] || { echo "soak: missing or non-executable: $b" >&2; exit 1; }
done

mkdir -p "$OUT/fail" || exit 1
: > "$OUT/progress"
: > "$OUT/summary"

frel=0; fstr=0; nrel=0; nstr=0
start=$(date +%s)

r=1
while [ "$r" -le "$ROUNDS" ]; do
    pids=""
    i=1
    while [ "$i" -le "$PAR" ]; do
        # Inline subshells with explicit PIDs, not backgrounded shell
        # functions: the function form left `wait` blocked on subshells with no
        # children left, wedging the soak twice.
        ( timeout "$PER_RUN_TIMEOUT" "$REL" > "$OUT/.out_rel_$i" 2>&1
          echo $? > "$OUT/.rc_rel_$i" ) &
        pids="$pids $!"
        ( timeout "$PER_RUN_TIMEOUT" "$STRICT" > "$OUT/.out_strict_$i" 2>&1
          echo $? > "$OUT/.rc_strict_$i" ) &
        pids="$pids $!"
        i=$((i+1))
    done
    for p in $pids; do wait "$p"; done

    i=1
    while [ "$i" -le "$PAR" ]; do
        for tag in rel strict; do
            rc=$(cat "$OUT/.rc_${tag}_$i" 2>/dev/null)
            [ -z "$rc" ] && rc="missing"
            if [ "$tag" = rel ]; then nrel=$((nrel+1)); else nstr=$((nstr+1)); fi
            if [ "$rc" != "0" ]; then
                if [ "$tag" = rel ]; then frel=$((frel+1)); else fstr=$((fstr+1)); fi
                cp "$OUT/.out_${tag}_$i" \
                   "$OUT/fail/${tag}_r${r}_i${i}_rc${rc}.txt" 2>/dev/null
            fi
        done
        i=$((i+1))
    done

    now=$(date +%s)
    echo "round $r/$ROUNDS  elapsed $((now-start))s  rel $frel/$nrel  strict $fstr/$nstr" \
        > "$OUT/progress"
    r=$((r+1))
done

echo "FINAL rel $frel/$nrel  strict $fstr/$nstr" > "$OUT/summary"
