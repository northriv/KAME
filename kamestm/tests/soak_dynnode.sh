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
# NARROWED (2026-08-22) to: ILP32 *and* the pool allocator.  Either one alone
# is clean; both together fail.  All A/Bs below are same tree, same load, same
# source, one variable each, 4-way concurrent on i486/i586:
#
#     i486  pool ON / OFF          7-10 %  /  0 / 42
#     i586  pool ON / OFF         21-25 %  /  0 / 44
#     x86-64, arm64 (pool ON)      never fired
#
# i586 is what makes this precise: it is ILP32 but has CMPXCHG8B, so
# KAME_STM_COMPACT_STATE is 0 there — the full 64-bit stamp, the uint64_t
# BitmapWord, privilege enabled.  It fails MORE than i486, so compact mode is
# not the cause; pointer width is.  Excluded the same way:
#
#     link form   static 4/30 vs shared 6/30   -> not the DSO boundary.  (An
#                 earlier "static never reproduces" reading was wrong: that
#                 hand build was missing -fPIC / -fvisibility-inlines-hidden /
#                 -fno-semantic-interposition, so it was a different binary.)
#     TLS model   initial-exec 13/80 vs default 21/80 -> shifts the rate, not
#                 the cause.
#     -O level    -O3 gives SIGSEGV, -O1 gives hangs; both fire.
#
# And it is specific to THIS test, at equal 4-way concurrency and equal wall
# time per arm: dynamic_node 2/36, while transaction_test 0/588,
# negotiation 0/312, payload_integrity_mixed 0/468, reanchor 0/36988.  What
# only dynamic_node does is change STRUCTURE under contention (insert /
# release / swap in four threads plus the main thread).  Below it, the
# components are clean on their own: atomic_shared_ptr/scoped/queue/intrusive
# 0/1600, the pool's own stress tests 0/128, the whole 46-test suite 0 (bar
# transaction_wait_budget_test, which measures latency budgets and cannot
# survive a saturated box — exclude it from load runs).
#
# Crash sites cluster on the recursive bundle build:
# snapshot -> bundle -> bundle_subpacket -> bundle, from where they land in
# PacketWrapper::bundledBy (reading a recycled wrapper: every field garbage,
# m_bundledBy.m_ref == 0x1), in reverseLookup, or inside operator new itself.
# The last one matters: the heap's own structures are already corrupt, so this
# is not one stale pointer, it is broad damage.
#
# Ruled out inside the allocator: the 32-bit radix path.  `radix_lookup_slow`
# does drop its bound check on ILP32 (kBoundShift 48 >= 32), but the indices
# cannot escape — region_idx = up >> 25 caps at 127, so l1 is always 0 and l2
# always < 2048.  The comment there is correct; no OOB, no over-width shift.
#
# The wider soak had already shaped the suspicion.  Of 12 captured failures the
# largest group was rc=124 — a HANG hitting the per-run timeout — with SIGSEGV
# and SIGABRT behind it, and four of them printed `succeeded` FIRST and then
# died or wedged.  A fault after every check has passed is teardown, and the
# pool's teardown is where the thread-exit orphan-chain reclaim, the
# pthread_key destructors and static destruction order all meet.  It also
# explains why a hand-built binary that compiled allocator.cpp straight in
# never reproduced once, while the shared-library build did: the failing path
# runs across the DSO boundary.
#
# The one STM-level message seen, `STM lookup failed: payload ... not reachable`
# ending in an uncaught std::domain_error, is consistent with the same cause —
# an allocator that hands back memory it should not have — and should be
# re-judged only after the allocator side is fixed.  Worth noting separately
# that the STM throws that domain_error where nobody catches it, so a detected
# inconsistency takes the whole process down.
#
# Not yet done: a backtrace.  A -O3 -g rebuild under gdb had not caught the
# SIGSEGV within 14 rounds when the investigation was stopped; the crash is
# ~17% per run without gdb, so a longer gdb loop should get one.
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
