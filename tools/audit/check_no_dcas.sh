#!/bin/sh
# kamepoolalloc + kamestm: guard the "no unguarded 64-bit atomic" invariant.
#
# Both libraries deliberately support hosts where `atomic<uint64_t>` is NOT
# lock-free (ATOMIC_LLONG_LOCK_FREE != 2): RV32 (A extension gives 32-bit AMO
# only), ARMv5/v6, MIPS32, PPC32 — and i386/i486.  On those, a 64-bit atomic
# lowers to a LOCKED libatomic call, so one in a free path or a negotiation
# path silently destroys the lock-freedom the whole design rests on.
# `allocator_prv.h` keeps a uint32_t `BitmapWord` fallback for exactly this,
# `allocator.cpp` keeps its counters pointer-width (`rt_counter_t`), and
# kamestm has the whole KAME_STM_COMPACT_STATE layout plus `diag_counter_t`.
#
# None of that is checked by a normal 64-bit build — on x86-64 a stray
# `atomic<unsigned long long>` is lock-free and links fine.  This script is
# what makes those paths fail loudly instead of rotting.  It caught the §75 RT
# counters, which had put a CMPXCHG8B loop into `deallocate_chunk`, and then
# kamestm's always-on diagnostic counters, which broke the i486 link outright
# despite COMPACT_STATE being designed for that very target.
#
#   Phase 1  uint32_t BitmapWord fallback — compiles natively, no multilib.
#   Phase 2  i486 probe — CMPXCHG8B is i586/Pentium+, so -march=i486 lowers
#            every 64-bit atomic to __atomic_*_8.  Compile-only + symbol
#            scan (~10 s); needs gcc-multilib, skipped with a notice if absent.
#   Phase 3  the same i486 probe against the kamestm STM core, instantiated
#            through a minimal Node<> so the templates are actually emitted.
#
# Scope note: this guards the LIBRARIES, not their tests.  A test or bench may
# legitimately want a 64-bit tally (latency sums, progress counters) and so may
# not build on a no-DCAS host — same call as `g_alloc_size_histo`, which stays
# 64-bit because it is bumped per allocation.  Library code has no such excuse.
#
# Usage: tools/audit/check_no_dcas.sh          (from anywhere)
# Exit 1 on any regression; a skipped phase is NOT a failure.

cd "$(dirname "$0")/../.." || exit 1
status=0
CXX=${CXX:-g++}
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT

# ---- Phase 1: the uint32_t BitmapWord fallback still compiles --------------
printf 'no-dcas: [1/3] uint32_t BitmapWord fallback ... '
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

# i486_probe <tag> <hint> <src> <extra cxxflags...>
#   Compiles <src> for i486 and fails if the object references __atomic_*_8.
#   Echoes its own ok / FAILED / SKIPPED line; sets `status` on failure.
i486_probe() {
    tag=$1; hint=$2; src=$3; shift 3
    if ! "$CXX" -m32 -E -x c++ /dev/null >/dev/null 2>&1; then
        echo "SKIPPED (no 32-bit multilib; install gcc-multilib g++-multilib)"
        return
    fi
    # Pre-probe what -m32 -march=i486 ACTUALLY targets before compiling the
    # real source.  Apple clang accepts the flags and emits 32-bit ARMv4T,
    # which used to surface in two different ways: allocator.cpp "compiled"
    # to an empty object (caught below), while the STM probe FAILS to compile
    # outright — ARMv4T has no lock-free atomics at all, so atomic.h's
    # int_cas_max fallthrough leaves the typedef undefined and the phase
    # reported a failure that no real i486 toolchain would produce
    # (ATOMIC_INT_LOCK_FREE == 2 there; CMPXCHG is i486+).  One trivial TU
    # settles the toolchain question for both phases.
    if [ -z "${NO_DCAS_ARCH_OK+x}" ]; then
        echo 'int kame_no_dcas_arch_probe;' > "$tmp/archprobe.c"
        if "$CXX" -m32 -march=i486 -c "$tmp/archprobe.c" \
                -o "$tmp/archprobe.o" 2>/dev/null \
                && file "$tmp/archprobe.o" 2>/dev/null \
                       | grep -qiE 'Intel (80)?386|i386|80486|x86'; then
            NO_DCAS_ARCH_OK=1
        else
            NO_DCAS_ARCH_OK=0
        fi
    fi
    if [ "$NO_DCAS_ARCH_OK" != "1" ]; then
        echo "SKIPPED (toolchain accepted -m32 -march=i486 but targets"
        echo "         $( (file "$tmp/archprobe.o" 2>/dev/null || echo 'nothing') | sed 's/.*: //')"
        echo "         — needs a real i386 multilib toolchain, e.g. Linux g++ -m32)"
        return
    fi
    if ! "$CXX" -m32 -march=i486 -std=c++17 -O2 "$@" -c "$src" \
            -o "$tmp/$tag.o" 2>"$tmp/$tag.log"; then
        echo "FAILED (compile)"
        sed 's/^/    /' "$tmp/$tag.log" | head -20
        status=1
        return
    fi
    if [ -z "$(nm "$tmp/$tag.o" 2>/dev/null | sed -n 's/^.* T \(.*\)$/\1/p')" ]; then
        # The compile "succeeded" but produced no code, so the symbol scan
        # below would find no __atomic_*_8 no matter what the source did — a
        # green result that proves nothing, which is worse than an honest skip.
        #
        # This is not hypothetical: on arm64 macOS the default CXX is Apple
        # clang, `-m32 -E` succeeds (preprocessing does not care about the
        # target), and `-m32 -march=i486 -c allocator.cpp` then exits 0 having
        # emitted a 176-byte `Mach-O object arm_v4t` with no symbol table at
        # all — clang read the flags as 32-bit ARMv4T.  The probe reported
        # "ok" on the primary dev platform while checking nothing.
        echo "SKIPPED (toolchain accepted -m32 -march=i486 but emitted no code for"
        echo "         that target: $( (file "$tmp/$tag.o" 2>/dev/null || echo '?') | sed 's/.*: //')"
        echo "         — needs a real i386 multilib toolchain, e.g. Linux g++ -m32)"
        return
    fi
    # `nm` prints undefined symbols as "U <name>".  Any __atomic_*_8 here is a
    # 64-bit atomic the CPU cannot do in one instruction.
    found=$(nm "$tmp/$tag.o" | sed -n 's/.* U \(__atomic_[a-z_]*_8\)$/\1/p' | sort -u)
    if [ -n "$found" ]; then
        echo "FAILED"
        echo "  64-bit atomic(s) lowered to LOCKED libatomic calls:"
        echo "$found" | sed 's/^/    /'
        echo "  A locked op here breaks lock-freedom on every no-DCAS target"
        echo "  (RV32, ARMv6, MIPS32, i486).  $hint"
        status=1
    else
        echo "ok"
    fi
}

# ---- Phase 2: no 64-bit atomic reaches an allocator hot path ---------------
printf 'no-dcas: [2/3] i486 probe: kamepoolalloc ... '
i486_probe alloc \
    "Give the offending counter pointer width (see rt_counter_t), or guard it on ATOMIC_LLONG_LOCK_FREE the way BitmapWord does." \
    kamepoolalloc/allocator.cpp -DKAMEPOOLALLOC_DYLIB -I kamepoolalloc

# ---- Phase 3: same, for the kamestm STM core ------------------------------
# transaction_impl.h holds the out-of-line definitions and is included by ONE
# TU per program (kame/xnode.cpp in production, each test otherwise), so the
# probe has to build such a TU itself.  A minimal Node<> subclass instantiates
# the negotiation templates; without it the headers emit almost nothing and
# the scan would be vacuous in the same way the arm_v4t case above is.
cat > "$tmp/stm_probe.cpp" <<'PROBE_EOF'
// support_standalone.h is the tests' Qt-free stand-in for kame/support.h
// (XKameError etc.); the STM headers need one or the other.  Using the test
// one keeps the probe independent of the Qt build.
#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
class ProbeNode : public Transactional::Node<ProbeNode> {
public:
    struct Payload : public Transactional::Node<ProbeNode>::Payload { long m_x = 0; };
};
// Force emission of the commit / negotiate paths the counters live in.
extern "C" void kame_no_dcas_probe(ProbeNode *n) {
    n->iterate_commit([=](Transactional::Transaction<ProbeNode> &tr){ tr[ *n].m_x++; });
    Transactional::Snapshot<ProbeNode> shot( *n);
    (void)shot[ *n].m_x;
}
PROBE_EOF
printf 'no-dcas: [3/3] i486 probe: kamestm ... '
i486_probe stm \
    "kamestm keeps KAME_STM_COMPACT_STATE for these hosts — an always-on counter must use diag_counter_t, and anything wider must sit behind a KAME_ENABLE_* gate that compact mode forces off." \
    "$tmp/stm_probe.cpp" -I kamestm -I kamestm/tests -I kamepoolalloc -I .

exit $status
