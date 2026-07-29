#!/bin/sh
# kamepoolalloc: guard the "no unguarded 64-bit atomic" invariant.
#
# The allocator deliberately supports hosts where `atomic<uint64_t>` is NOT
# lock-free (ATOMIC_LLONG_LOCK_FREE != 2): RV32 (A extension gives 32-bit AMO
# only), ARMv5/v6, MIPS32, PPC32 — and i386/i486.  On those, a 64-bit atomic
# lowers to a LOCKED libatomic call, so one in a free path silently destroys
# the lock-freedom the whole design rests on.  `allocator_prv.h` keeps a
# uint32_t `BitmapWord` fallback for exactly this, and `allocator.cpp` keeps
# its counters pointer-width for the same reason.
#
# Neither guarantee is checked by a normal 64-bit build — on x86-64 a stray
# `atomic<unsigned long long>` is lock-free and links fine.  This script is
# what makes the two paths fail loudly instead of rotting.  It caught the §75
# RT counters, which had put a CMPXCHG8B loop into `deallocate_chunk`.
#
#   Phase 1  uint32_t BitmapWord fallback — compiles natively, no multilib.
#   Phase 2  i486 probe — CMPXCHG8B is i586/Pentium+, so -march=i486 lowers
#            every 64-bit atomic to __atomic_*_8.  Compile-only + symbol
#            scan (~10 s); needs gcc-multilib, skipped with a notice if absent.
#
# Usage: tools/audit/check_no_dcas.sh          (from anywhere)
# Exit 1 on any regression; a skipped Phase 2 is NOT a failure.

cd "$(dirname "$0")/../.." || exit 1
status=0
CXX=${CXX:-g++}
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT

# ---- Phase 1: the uint32_t BitmapWord fallback still compiles --------------
printf 'no-dcas: [1/2] uint32_t BitmapWord fallback ... '
if "$CXX" -std=c++17 -O2 -DKAMEPOOLALLOC_DYLIB -DKAME_FORCE_UINT32_BITMAP \
        -I kamepoolalloc -c kamepoolalloc/allocator.cpp -o "$tmp/u32.o" \
        2>"$tmp/u32.log"; then
    echo "ok"
else
    echo "FAILED"
    echo "  the KAME_FORCE_UINT32_BITMAP path no longer compiles:"
    sed 's/^/    /' "$tmp/u32.log" | head -20
    status=1
fi

# ---- Phase 2: no 64-bit atomic reaches a hot path --------------------------
printf 'no-dcas: [2/2] i486 (no DCAS) 64-bit atomic probe ... '
if ! "$CXX" -m32 -E -x c++ /dev/null >/dev/null 2>&1; then
    echo "SKIPPED (no 32-bit multilib; install gcc-multilib g++-multilib)"
elif ! "$CXX" -m32 -march=i486 -std=c++17 -O2 -DKAMEPOOLALLOC_DYLIB \
        -I kamepoolalloc -c kamepoolalloc/allocator.cpp -o "$tmp/i486.o" \
        2>"$tmp/i486.log"; then
    echo "FAILED (compile)"
    sed 's/^/    /' "$tmp/i486.log" | head -20
    status=1
elif [ -z "$(nm "$tmp/i486.o" 2>/dev/null | sed -n 's/^.* T \(.*\)$/\1/p')" ]; then
    # The compile "succeeded" but produced no code, so the symbol scan below
    # would find no __atomic_*_8 no matter what the source did — a green result
    # that proves nothing, which is worse than an honest skip.
    #
    # This is not hypothetical: on arm64 macOS the default CXX is Apple clang,
    # `-m32 -E` succeeds (preprocessing does not care about the target), and
    # `-m32 -march=i486 -c allocator.cpp` then exits 0 having emitted a
    # 176-byte `Mach-O object arm_v4t` with no symbol table at all — clang read
    # the flags as 32-bit ARMv4T.  Phase 2 reported "ok" on the primary dev
    # platform while checking nothing.
    echo "SKIPPED (toolchain accepted -m32 -march=i486 but emitted no code for"
    echo "         that target: $( (file "$tmp/i486.o" 2>/dev/null || echo '?') | sed 's/.*: //')"
    echo "         — needs a real i386 multilib toolchain, e.g. Linux g++ -m32)"
else
    # `nm` prints undefined symbols as "U <name>".  Any __atomic_*_8 here is a
    # 64-bit atomic the CPU cannot do in one instruction.
    found=$(nm "$tmp/i486.o" | sed -n 's/.* U \(__atomic_[a-z_]*_8\)$/\1/p' | sort -u)
    if [ -n "$found" ]; then
        echo "FAILED"
        echo "  64-bit atomic(s) lowered to LOCKED libatomic calls:"
        echo "$found" | sed 's/^/    /'
        echo "  A locked op in an allocator free path breaks lock-freedom on"
        echo "  every no-DCAS target (RV32, ARMv6, MIPS32, i486).  Give the"
        echo "  offending counter pointer width (see rt_counter_t), or guard"
        echo "  it on ATOMIC_LLONG_LOCK_FREE the way BitmapWord does."
        status=1
    else
        echo "ok"
    fi
fi

exit $status
