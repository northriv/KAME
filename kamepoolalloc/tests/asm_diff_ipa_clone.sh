#!/bin/sh
# §13.17 — one-command asm diff of allocator.cpp: -O2 vs -O2 -fipa-cp-clone.
# The 19/35 minimal pair (handoff §6): adding ONE pass to -O2 reproduces the
# fault, so the codegen delta of interest is exactly this diff -- far smaller
# than -O2 vs -O3.  Produces per-symbol disassembly for both arms, a list of
# clone symbols (.constprop.*), a list of changed base symbols, and unified
# diffs for every changed/cloned family.  Read the NC7 families first
# (bucket_release_chunk, find_training_zeros callers, ... -- record the full
# list in DYNNODE_UAF_HANDOFF.md when known).
#
# Usage: ./asm_diff_ipa_clone.sh [out_dir] [CXX] [extra flags...]
#   e.g. ./asm_diff_ipa_clone.sh /tmp/asmdiff g++-15 -DKAME_POISON_FORENSIC
# Arms default to the minimal pair; override for other comparisons:
#   ARM_A_FLAGS="-O2" ARM_B_FLAGS="-O3" ./asm_diff_ipa_clone.sh ...
set -e
OUT=${1:-/tmp/asmdiff}; shift 2>/dev/null || true
CXX=${1:-g++}; shift 2>/dev/null || true
SRC=$(dirname "$0")/../allocator.cpp
BASEFLAGS="-std=gnu++17 -fPIC -DNDEBUG -fno-omit-frame-pointer -ffunction-sections -c"
mkdir -p "$OUT/A" "$OUT/B"

ARM_A_FLAGS=${ARM_A_FLAGS:--O2}
ARM_B_FLAGS=${ARM_B_FLAGS:--O2 -fipa-cp-clone}
$CXX $BASEFLAGS $ARM_A_FLAGS $* "$SRC" -o "$OUT/A/allocator.o"
$CXX $BASEFLAGS $ARM_B_FLAGS $* "$SRC" -o "$OUT/B/allocator.o"

split_syms() {  # $1 = arm dir
    objdump -d --no-show-raw-insn --no-addresses "$1/allocator.o" 2>/dev/null \
      || objdump -d --no-show-raw-insn "$1/allocator.o" | sed 's/^ *[0-9a-f]*:\t//'
}
for arm in A B; do
    split_syms "$OUT/$arm" | awk -v dir="$OUT/$arm/sym" '
        BEGIN { system("mkdir -p " dir) }
        /^[0-9a-f]* <.*>:$/ || /^<.*>:$/ {
            sym = $0; gsub(/^[0-9a-f]* </, "", sym); gsub(/^</, "", sym)
            gsub(/>:$/, "", sym); gsub(/[^A-Za-z0-9_.]/, "_", sym)
            if (length(sym) > 200) sym = substr(sym, 1, 200)
            f = dir "/" sym ".s"; next
        }
        f { print > f }'
done

: > "$OUT/summary.txt"
echo "== clone symbols (only in B) ==" >> "$OUT/summary.txt"
for f in "$OUT"/B/sym/*.s; do
    b=$(basename "$f")
    [ -f "$OUT/A/sym/$b" ] || echo "  $b" >> "$OUT/summary.txt"
done
echo "== changed symbols (present in both, body differs) ==" >> "$OUT/summary.txt"
mkdir -p "$OUT/diff"
for f in "$OUT"/A/sym/*.s; do
    b=$(basename "$f")
    if [ -f "$OUT/B/sym/$b" ] && ! cmp -s "$f" "$OUT/B/sym/$b"; then
        echo "  $b" >> "$OUT/summary.txt"
        diff -u "$f" "$OUT/B/sym/$b" > "$OUT/diff/$b.diff" || true
    fi
done
echo "wrote $OUT/summary.txt ; per-symbol diffs in $OUT/diff/ ;" \
     "clone bodies in $OUT/B/sym/*constprop*"
