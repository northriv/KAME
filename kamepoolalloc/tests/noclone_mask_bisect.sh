#!/bin/bash
# §13.121  SUBTRACTIVE bisect from a FIRING baseline.
#
# §13.120 showed per-function licensing cannot reach the four functions
# -fipa-cp-clone specialises from caller context (l1_pop_fit, global_pop_fit,
# recycle_pop_fit, acquire_tag_ref_).  This runs the other direction: build with
# the flag ON (the 9/10 configuration) and take cloning away from one candidate
# at a time.
#
#   ./noclone_mask_bisect.sh [masks...]        default: 0 0x2 0x7 asp 0x7+asp
#
# It reports the ipa-cp DECISION counts, not surviving .constprop symbols --
# §13.120's whole lesson.  An arm whose decision count did not drop suppressed
# nothing and is VACUOUS, not a negative.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-g++}
OUT=${OUT:-/tmp/noclone_mask}; mkdir -p "$OUT"
BASEFLAGS="-O2 -fipa-cp-clone -std=c++17 -pthread -fPIC -shared
           -DKAMEPOOLALLOC_DYLIB -DKAME_POISON_FORENSIC -DUSE_KAME_ALLOCATOR"
ARMS=${*:-"0 0x2 0x7 asp both"}

decisions() {   # $1 = dump file; prints "<nodes> <functions>"
    local n f
    n=$(grep -c 'Creating a specialized node' "$1" 2>/dev/null || echo 0)
    f=$(grep 'Creating a specialized node' "$1" 2>/dev/null |
        sed 's/.*node of //; s/\.constprop.*//' | sort -u | wc -l)
    echo "$n $f"
}

for a in $ARMS; do
    case $a in
      asp)  def="-DKAME_NOCLONE_MASK=0 -DKAME_ASP_NOCLONE" ;;
      both) def="-DKAME_NOCLONE_MASK=0x7 -DKAME_ASP_NOCLONE" ;;
      *)    def="-DKAME_NOCLONE_MASK=$a" ;;
    esac
    so="$OUT/kp_$a.so"; dump="$OUT/kp_$a.dump"
    rm -f "$so" "$dump"
    # -fdump-ipa-cp-details writes <basename>.<pass>.ipa-cp next to the output;
    # -fdump-file-prefix keeps arms from overwriting each other.
    ( cd "$OUT" && $CXX $BASEFLAGS $def -fdump-ipa-cp-details \
        -I"$SRC" "$SRC/allocator.cpp" -o "$so" ) 2>"$OUT/build_$a.log"
    if [ ! -f "$so" ]; then
        echo "arm=$a BUILD FAILED -- see $OUT/build_$a.log"   # never /dev/null (§13.119)
        continue
    fi
    cat "$OUT"/allocator.cpp.*ipa-cp* > "$dump" 2>/dev/null
    rm -f "$OUT"/allocator.cpp.*ipa-cp*
    read nodes funcs < <(decisions "$dump")
    echo "arm=$a  specialized nodes=$nodes functions=$funcs  so=$so"
    grep 'Creating a specialized node' "$dump" 2>/dev/null |
        sed 's/.*node of //; s/\.constprop.*//' | sort | uniq -c | sort -rn |
        sed 's/^/    /'
done
echo
echo "Compare each arm's function list against arm 0's: an arm that did not"
echo "remove its target from the list suppressed nothing (VACUOUS)."
echo "Then run the reproducer against each .so, interleaved in ONE job."
