#!/bin/sh
# Long soak for the transaction_dynamic_node_test failure seen once on i486
# (2026-08-22).  The point is NOT throughput: it is that when the failure
# recurs, the evidence survives.  The original occurrence was lost because
# ctest overwrote Testing/Temporary/LastTest.log on the next run.
#
# Two arms, both i486 (the configuration that failed):
#   rel    — Release, the exact build that failed.  Best chance of
#            reproducing the original signature.
#   strict — -UNDEBUG -DTRANSACTIONAL_STRICT_assert -O1.  30 plain asserts
#            plus 12 STRICT_assert packet-consistency checks are live, and
#            KAME_STM_HANG_ABORT_N reverts from 0 to 3, so a livelock aborts
#            with a [HANG] dump instead of hanging silently.  Best chance of
#            localising the fault at the point it happens rather than at the
#            end-of-test tally.
#
# Every non-zero exit is kept verbatim under $OUT/fail/, with the arm, round
# and iteration in the filename.  Progress is a single line in $OUT/progress
# so it can be polled without disturbing the run.
#
# Usage:  soak_dynnode.sh [rounds] [parallel-per-arm] [outdir]
# Detach: setsid nohup soak_dynnode.sh 500 2 /tmp/soak > /dev/null 2>&1 &
#   (setsid matters — a plain `&` job died with its parent shell twice while
#    this failure was being chased.)

ROUNDS=${1:-200}
PAR=${2:-2}
OUT=${3:-/tmp/soak_dynnode}

REL=/tmp/bt-486/kamestm-tests/transaction_dynamic_node_test
STRICT=/tmp/bt-486s/kamestm-tests/transaction_dynamic_node_test

mkdir -p "$OUT/fail" || exit 1
: > "$OUT/progress"
: > "$OUT/summary"

frel=0; fstr=0; nrel=0; nstr=0
start=$(date +%s)

run_one() {   # $1=binary $2=tag $3=round $4=idx
    o="$OUT/.tmp_$2_$4"
    "$1" > "$o" 2>&1
    rc=$?
    echo "$rc" > "$OUT/.rc_$2_$4"
    if [ "$rc" != "0" ]; then
        cp "$o" "$OUT/fail/${2}_r${3}_i${4}_rc${rc}.txt"
    fi
}

r=1
while [ "$r" -le "$ROUNDS" ]; do
    i=1
    while [ "$i" -le "$PAR" ]; do
        run_one "$REL"    rel    "$r" "$i" &
        run_one "$STRICT" strict "$r" "$i" &
        i=$((i+1))
    done
    wait

    i=1
    while [ "$i" -le "$PAR" ]; do
        nrel=$((nrel+1))
        [ "$(cat "$OUT/.rc_rel_$i" 2>/dev/null)" != "0" ] && frel=$((frel+1))
        nstr=$((nstr+1))
        [ "$(cat "$OUT/.rc_strict_$i" 2>/dev/null)" != "0" ] && fstr=$((fstr+1))
        i=$((i+1))
    done

    now=$(date +%s)
    echo "round $r/$ROUNDS  elapsed $((now-start))s  rel $frel/$nrel  strict $fstr/$nstr" \
        > "$OUT/progress"
    r=$((r+1))
done

echo "FINAL rel $frel/$nrel  strict $fstr/$nstr" >> "$OUT/summary"
cat "$OUT/progress" >> "$OUT/summary"
