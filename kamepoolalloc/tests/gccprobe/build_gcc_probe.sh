#!/bin/bash
# §13.145  Build allocator.cpp with REAL GCC on macOS, for CODEGEN INSPECTION
# ONLY.  Objects produced here must never be run or linked into anything that
# runs -- see the shim header for why.
#
# This exists because every static check in this investigation has been waiting
# on the Linux session, and none of them needs to execute anything: a `.o` is
# enough for objdump, nm, and -fdump-ipa-cp-details.  Three things block GCC on
# macOS, and none is a project rule:
#
#  1. The macOS 26 SDK's <malloc/malloc.h> pulls in <mach/message.h>, which uses
#     the clang-only `xnu_static_assert_struct_size` and GCC rejects outright.
#     `malloc_shim.h` (placed as <malloc/malloc.h> earlier on the include path)
#     declares the six zone functions and the one struct member the code touches.
#     The struct layout is plausible, not faithful -- hence inspection only.
#  2. Darwin GCC does not support `__attribute__((constructor(N)))` priorities.
#     A scratch copy has `constructor(N)` rewritten to `constructor`; that
#     changes initialisation ORDER, not any function body being censused.
#  3. Nothing else.  That is the whole of "kamepoolalloc refuses to build under
#     GCC on macOS" (cdb70d2cf) for this purpose.
#
# Usage: build_gcc_probe.sh [<gcc>] ; writes /tmp/g15_{base,clone}.o and prints
# the IPA-CP decision counts for the §13.119 minimal pair.
set -u
GXX=${1:-/opt/local/bin/g++-mp-15}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)                 # kamepoolalloc/
OUT=${OUT:-/tmp/gccprobe}
mkdir -p "$OUT/malloc"
cp "$HERE/malloc_shim.h" "$OUT/malloc/malloc.h"
sed 's/constructor(\([0-9]*\))/constructor/g' "$SRC/allocator.cpp" > "$OUT/allocator.cpp"

for arm in "base:-O2" "clone:-O2 -fipa-cp-clone"; do
    name=${arm%%:*}; flags=${arm#*:}
    rm -f "$OUT"/allocator.cpp.*ipa-cp*
    ( cd "$OUT" && $GXX -std=c++17 $flags -pthread -c \
        -DKAMEPOOLALLOC_DYLIB -DUSE_KAME_ALLOCATOR \
        -I"$OUT" -I"$SRC" -fdump-ipa-cp-details \
        allocator.cpp -o "/tmp/g15_$name.o" ) || { echo "arm=$name BUILD FAILED"; continue; }
    printf "%-6s constprop syms=%-4s size=%s\n" "$name" \
        "$(nm -C /tmp/g15_$name.o | grep -c constprop)" \
        "$(wc -c < /tmp/g15_$name.o | tr -d ' ')"
done
echo
echo "Now:  python3 $SRC/tests/tagmask_census.py /tmp/g15_base.o /tmp/g15_clone.o --all"
echo "Caveat: the target is the HOST arch.  On an arm64 Mac these are aarch64"
echo "bodies, while the firing build is x86-64 -- the middle-end question (was"
echo "the operation kept?) travels; the instruction selection does not."
