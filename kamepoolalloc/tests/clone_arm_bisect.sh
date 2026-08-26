#!/bin/bash
# §13.112  POSITIVE clone bisect.  Baseline is plain -O2 (0/8 in §6); each arm
# licenses -fipa-cp-clone for exactly ONE allocator function.  An arm that
# FIRES names that function; §5's objection to the subtractive direction
# (noclone on one function moved 72 others) does not apply to a build that
# starts from a clean baseline and adds one licence.
#
#   ./clone_arm_bisect.sh <reproducer> [runs-per-arm] [arms...]
#
# For each arm it (1) builds the pool at -O2 with that arm licensed,
# (2) VERIFIES a .constprop clone actually appeared -- -O2's IPA-CP
# profitability thresholds may refuse it, and an arm with no clone is vacuous,
# not negative (§13.61) -- and (3) runs the reproducer.
set -u
REPRO=${1:?usage: clone_arm_bisect.sh <reproducer-binary-builder> [runs] [arms...]}
RUNS=${2:-16}; shift 2 || true
ARMS=${*:-"1 2 3 4 5 6 7"}
SRC=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-g++}
OUT=${OUT:-/tmp/clone_arm}
mkdir -p "$OUT"

# Reference: what does -O2 -fipa-cp-clone clone GLOBALLY?  This is the
# measured clone set, and the arm list should be checked against it -- the
# notebook never recorded NC7's full membership.
$CXX -O2 -fipa-cp-clone -std=c++17 -pthread -c \
    -DKAMEPOOLALLOC_DYLIB -DKAME_POISON_FORENSIC -DUSE_KAME_ALLOCATOR \
    -I"$SRC" "$SRC/allocator.cpp" -o "$OUT/ref.o" 2>/dev/null
echo "=== global clone set at -O2 -fipa-cp-clone ==="
nm -C "$OUT/ref.o" | grep -o '[A-Za-z_][A-Za-z0-9_:]*\.constprop[0-9.]*' \
    | sed 's/\.constprop.*//' | sort | uniq -c | sort -rn
echo

for a in $ARMS; do
    obj="$OUT/arm$a.o"
    $CXX -O2 -std=c++17 -pthread -c -DKAME_CLONE_ARM=$a \
        -DKAMEPOOLALLOC_DYLIB -DKAME_POISON_FORENSIC -DUSE_KAME_ALLOCATOR \
        -I"$SRC" "$SRC/allocator.cpp" -o "$obj" 2>"$OUT/arm$a.buildlog" || {
        echo "arm=$a BUILD FAILED (see $OUT/arm$a.buildlog)"; continue; }
    n=$(nm -C "$obj" | grep -c '\.constprop')
    if [ "$n" -eq 0 ]; then
        echo "arm=$a  VACUOUS -- licence granted but -O2 produced no clone; not a negative result"
        continue
    fi
    echo "arm=$a  clones=$n  -> running $RUNS x"
    fails=0
    for i in $(seq 1 "$RUNS"); do
        "$REPRO" "$obj" >>"$OUT/arm$a.runlog" 2>&1 || fails=$((fails+1))
    done
    echo "arm=$a  FAILURES $fails/$RUNS"
done
