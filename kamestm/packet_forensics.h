/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the file COPYING and AUTHORS.
***************************************************************************/
//! \file packet_forensics.h
//! Debug-only detector for the `transaction_dynamic_node_test` fault:
//! a `Node<XN>::Packet` destroyed while a reachable `PacketWrapper` still
//! holds `m_packet` pointing at it (see kamestm/tests/DYNNODE_UAF_HANDOFF.md).
//!
//! Built only when `KAME_STM_PACKET_FORENSICS` is defined.  It replaces
//! `Packet`'s class-level `operator new`/`delete` (legal because `Packet`
//! derives from the *intrusive* `atomic_countable`, so `local_shared_ptr`
//! allocates it with a plain `new Packet`, not an emplaced control block):
//!
//!   - every block carries a header with a magic word, the destroying
//!     thread id, and the destroying call stack;
//!   - `operator delete` poisons the body and parks the block in a bounded
//!     quarantine ring instead of returning it, so a stale pointer reads
//!     the DEAD magic rather than whatever the allocator recycled into it;
//!   - `Packet`'s accessors check the magic, so a use-after-free aborts at
//!     the *first* stale dereference with BOTH stacks printed.
//!
//! Why this and not the allocator-side poison of the handoff: that one only
//! fires when the recycled contents happen to be dereferenced into a
//! non-canonical address, so most UAFs pass silently and the rate depends
//! on the allocator and on -fipa-cp-clone.  This fires on every occurrence
//! and works with any allocator, which is what makes the *releasing* site
//! findable rather than just the reading site.

#ifndef PACKET_FORENSICS_H
#define PACKET_FORENSICS_H

#ifdef KAME_STM_PACKET_FORENSICS

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#if defined __unix__ || defined __APPLE__
    #include <execinfo.h>
    #include <unistd.h>
    #define KAME_PF_HAVE_BACKTRACE 1
#else
    #define KAME_PF_HAVE_BACKTRACE 0
#endif

namespace Transactional { namespace detail {

//! Trailer word count captured for each of the birth / death stacks.
enum : int {PF_BT_DEPTH = 24};
//! Blocks parked before the oldest is really released.  Big enough that a
//! stale reference is essentially never re-validated by a recycled block,
//! small enough to bound the test at a few tens of MB.
enum : std::size_t {PF_QUARANTINE = 1u << 16};

enum : std::uint64_t {
    PF_MAGIC_ALIVE = 0x5041434B4C495645ull, // "PACKLIVE"
    PF_MAGIC_DEAD  = 0x5041434B44454144ull, // "PACKDEAD"
};

struct PacketForensicsHeader {
    std::uint64_t magic;
    std::size_t   size;
    const char   *type;         //!< which STM object this block held
    int           birth_tid;
    int           death_tid;
    int           birth_bt_n;
    int           death_bt_n;
    void         *birth_bt[PF_BT_DEPTH];
    void         *death_bt[PF_BT_DEPTH];
};

inline int pf_tid() noexcept {
    static std::atomic<int> next{0};
    static thread_local int id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

//! How much provenance each alloc/free records.  Selected by
//! `KAME_PF_TRACE`, because the cost difference is the difference between
//! a reproducer that still fires and one that does not:
//!   0 (default) — nothing.  A free costs a memset and two stores.
//!   1           — frame-pointer walk (needs -fno-omit-frame-pointer;
//!                 ~10 loads, keeps the race timing intact).
//!   2           — glibc backtrace().  Correct anywhere, but ~2 µs per
//!                 call, which is ~100x the cost of the allocation it
//!                 annotates and perturbs the interleaving out of range.
//! Capture at the *use* site (the abort path) always uses mode 2: it runs
//! exactly once, so accuracy is free there.
inline int pf_trace_mode() noexcept {
    static const int mode = []{
        const char *e = std::getenv("KAME_PF_TRACE");
        return e ? std::atoi(e) : 0;
    }();
    return mode;
}

//! Follow the saved-rbp chain.  Each frame is [saved rbp][return addr].
//! Bounded by a monotonicity + span check so a frameless (optimised-out)
//! frame terminates the walk instead of wandering off the stack.
inline int pf_capture_fp(void **bt) noexcept {
#if defined __x86_64__ || defined __i386__ || defined __aarch64__
    void **fp = static_cast<void **>(__builtin_frame_address(0));
    int n = 0;
    while(n < PF_BT_DEPTH && fp) {
        void **next = static_cast<void **>(fp[0]);
        // Frames grow downward: the next frame must be above this one and
        // within a sane distance, and aligned.
        if(next <= fp) break;
        if(reinterpret_cast<std::uintptr_t>(next)
            - reinterpret_cast<std::uintptr_t>(fp) > (1u << 20)) break;
        if(reinterpret_cast<std::uintptr_t>(next) & (sizeof(void *) - 1)) break;
        void *ra = fp[1];
        if( !ra) break;
        bt[n++] = ra;
        fp = next;
    }
    return n;
#else
    (void)bt; return 0;
#endif
}

inline int pf_capture(void **bt) noexcept {
    switch(pf_trace_mode()) {
    case 0: return 0;
    case 1: return pf_capture_fp(bt);
    default:
#if KAME_PF_HAVE_BACKTRACE
        return backtrace(bt, PF_BT_DEPTH);
#else
        return 0;
#endif
    }
}

//! Always-accurate variant for the one-shot abort path.
inline int pf_capture_exact(void **bt) noexcept {
#if KAME_PF_HAVE_BACKTRACE
    return backtrace(bt, PF_BT_DEPTH);
#else
    return pf_capture_fp(bt);
#endif
}

inline void pf_print(const char *what, void *const *bt, int n) noexcept {
    std::fprintf(stderr, "  %s stack (%d frames):\n", what, n);
    std::fflush(stderr);
#if KAME_PF_HAVE_BACKTRACE
    if(n > 0)
        backtrace_symbols_fd(const_cast<void **>(bt), n, 2);
#else
    (void)bt;
#endif
}

//! Backing allocator for the forensic blocks.  Default is the pool, so
//! `Packet` keeps being served by the allocator the fault needs and the
//! thread scaling is unchanged; `KAME_PF_MALLOC=1` switches to libc for a
//! control arm.  (A first cut used `malloc` unconditionally: with 16
//! threads and a shared quarantine that pinned the load average at ~1.3 on
//! a 4-core box — the arena locks had serialised the very concurrency the
//! reproducer depends on.)
extern "C" {
    void *kame_pool_malloc(std::size_t) __attribute__((weak));
    void  kame_pool_free(void *) __attribute__((weak));
}
inline bool pf_use_pool() noexcept {
    static const bool v = []{
        if( !&kame_pool_malloc || !&kame_pool_free) return false;
        const char *e = std::getenv("KAME_PF_MALLOC");
        return !(e && std::atoi(e));
    }();
    return v;
}
inline void *pf_raw_alloc(std::size_t n) noexcept {
    return pf_use_pool() ? kame_pool_malloc(n) : std::malloc(n);
}
inline void pf_raw_free(void *p) noexcept {
    if(pf_use_pool()) kame_pool_free(p); else std::free(p);
}

//! Per-thread FIFO of parked blocks — thread-local so the quarantine adds
//! no shared state of its own (a global ring's fetch_add plus the
//! cross-thread frees it produces were themselves a contention source).
struct PfRing {
    void **slot = nullptr;
    std::size_t pos = 0;
    ~PfRing() {
        // Thread exit: hand everything back.  The blocks are dead already;
        // holding them past the thread would only be a leak.
        if(slot) {
            for(std::size_t i = 0; i < PF_QUARANTINE; ++i)
                if(slot[i]) pf_raw_free(slot[i]);
            std::free(slot);
        }
    }
};
inline PfRing &pf_ring() noexcept {
    static thread_local PfRing r;
    if( !r.slot)
        r.slot = static_cast<void **>(std::calloc(PF_QUARANTINE, sizeof(void *)));
    return r;
}

//! Allocation / free tallies, reported at exit so a build that silently
//! bypasses these operators (the whole check would then be vacuous) is
//! visible as a pair of zeros rather than as a clean run.
inline std::atomic<std::size_t> &pf_allocs() noexcept {
    static std::atomic<std::size_t> n{0}; return n;
}
inline std::atomic<std::size_t> &pf_frees() noexcept {
    static std::atomic<std::size_t> n{0}; return n;
}
//! Blocks whose header carries neither magic.  A non-zero count means some
//! object of a watched type is allocated by a path that bypasses these
//! operators, so a "clean" run under-reports; it is printed rather than
//! treated as a fault, because the alternative (aborting) would turn an
//! instrumentation gap into a false positive.
inline std::atomic<std::size_t> &pf_unknown() noexcept {
    static std::atomic<std::size_t> n{0}; return n;
}
//! Quarantine on (default) or off (`KAME_PF_QUARANTINE=0`).
//!
//! Off matters more than it looks.  The record in DYNNODE_UAF_HANDOFF.md
//! says the fault needs freed addresses to come back into circulation --
//! delaying reuse by 2M allocations gave 0/41 against 13/41.  A quarantine
//! is exactly that delay, so a soak with it on may be a configuration that
//! structurally cannot fail, and a clean result would mean nothing.  With
//! it off the allocator's reuse pattern is intact (only the header's extra
//! bytes shift size classes) and the exclusivity check below -- the one
//! that separates an early free from a double allocation -- still works
//! unchanged, because it reads the header of the block being handed out,
//! not of one we are holding back.  What weakens is the DEAD check: a
//! recycled block's header may already belong to its next owner.  So run
//! quarantine ON to identify a use-after-free precisely, and OFF to ask
//! whether the fault happens at all under realistic reuse.
inline bool pf_quarantine() noexcept {
    static const bool v = []{
        const char *e = std::getenv("KAME_PF_QUARANTINE");
        return !(e && !std::atoi(e));
    }();
    return v;
}
inline bool pf_strict() noexcept {
    static const bool v = []{
        const char *e = std::getenv("KAME_PF_STRICT");
        return e && std::atoi(e);
    }();
    return v;
}
inline void pf_report_at_exit() noexcept {
    static bool once = (std::atexit([]{
        std::fprintf(stderr,
            "packet-forensics: %zu alloc, %zu free, %zu unknown-header reads\n",
            pf_allocs().load(), pf_frees().load(), pf_unknown().load());
    }), true);
    (void)once;
}

inline void *packet_forensics_new(std::size_t sz, const char *type) {
    pf_report_at_exit();
    pf_allocs().fetch_add(1, std::memory_order_relaxed);
    void *raw = pf_raw_alloc(sizeof(PacketForensicsHeader) + sz);
    if( !raw) throw std::bad_alloc();
    auto *h = static_cast<PacketForensicsHeader *>(raw);
    // Exclusivity.  The poison evidence in DYNNODE_UAF_HANDOFF.md cannot
    // tell "the STM dropped the last reference too early" from "the
    // allocator handed the same slot to two live objects" -- in both cases
    // a live object's storage reads back as someone else's poison.  This
    // separates them: a fresh block whose header still says ALIVE is one
    // our records show as in use, so the allocator returned a live block.
    // Costs one load per allocation.
    if(h->magic == PF_MAGIC_ALIVE) {
        std::fprintf(stderr,
            "\n*** ALLOCATOR RETURNED A LIVE BLOCK ***\n"
            "  %p (%zu bytes) is being handed to a %s on tid %d,\n"
            "  but our records say it still holds a live %s (tid %d)\n",
            static_cast<void *>(h + 1), sz, type, pf_tid(),
            h->type ? h->type : "?", h->birth_tid);
        pf_print("previous alloc", h->birth_bt, h->birth_bt_n);
        void *abt[PF_BT_DEPTH];
        pf_print("this alloc", abt, pf_capture_exact(abt));
        std::fflush(stderr);
        std::abort();
    }
    h->magic = PF_MAGIC_ALIVE;
    h->size = sz;
    h->type = type;
    h->birth_tid = pf_tid();
    h->death_tid = -1;
    h->birth_bt_n = pf_capture(h->birth_bt);
    h->death_bt_n = 0;
    return static_cast<void *>(h + 1);
}

inline void packet_forensics_delete(void *p) noexcept {
    if( !p) return;
    auto *h = static_cast<PacketForensicsHeader *>(p) - 1;
    if(h->magic == PF_MAGIC_DEAD) {
        std::fprintf(stderr,
            "\n*** STM DOUBLE FREE: %s at %p (first freed by tid %d) ***\n",
            h->type ? h->type : "?", p, h->death_tid);
        pf_print("first free", h->death_bt, h->death_bt_n);
        void *bt[PF_BT_DEPTH];
        pf_print("second free", bt, pf_capture_exact(bt));
        std::fflush(stderr);
        std::abort();
    }
    pf_frees().fetch_add(1, std::memory_order_relaxed);
    h->magic = PF_MAGIC_DEAD;
    h->death_tid = pf_tid();
    h->death_bt_n = pf_capture(h->death_bt);
    // Poison the body so a stale read of any member is obviously wrong even
    // if the magic check is bypassed.
    std::memset(p, 0xBD, h->size);

    if( !pf_quarantine()) { pf_raw_free(h); return; }
    PfRing &r = pf_ring();
    if( !r.slot) { pf_raw_free(h); return; }   // out of memory: no quarantine
    std::size_t i = r.pos++ % PF_QUARANTINE;
    void *old = r.slot[i];
    r.slot[i] = h;
    if(old) pf_raw_free(old);
}

//! \return silently when \a p is live; aborts with both stacks when its
//! header says the block was freed.  \a what names the type, \a where the
//! accessor, so the report says which object died and how it was reached.
inline void packet_forensics_check(const void *p, const char *what,
                                   const char *where) noexcept {
    if( !p) return;
    auto *h = static_cast<const PacketForensicsHeader *>(p) - 1;
    if(h->magic == PF_MAGIC_ALIVE) return;
    if(h->magic != PF_MAGIC_DEAD) {
        // Neither magic.  Two readings, and they are not equivalent:
        //   - an instrumentation gap (some object of this type is built by
        //     a path that bypasses our operators), or
        //   - the block's memory was handed to something else entirely --
        //     which is what an allocator returning a chunk while units in
        //     it are still live would look like, and that case is invisible
        //     to the DEAD check because the header never gets stamped.
        // A clean run measures exactly zero of these, so KAME_PF_STRICT=1
        // promotes the first one to a fault; the default only tallies them,
        // since aborting on a gap would be a false positive.
        pf_unknown().fetch_add(1, std::memory_order_relaxed);
        if( !pf_strict()) return;
        std::fprintf(stderr,
            "\n*** STM BLOCK NO LONGER ITSELF (KAME_PF_STRICT) ***\n"
            "  %s %p accessed via %s() by tid %d\n"
            "  header magic = 0x%016llx (neither ALIVE nor DEAD)\n",
            what, p, where, pf_tid(), (unsigned long long)h->magic);
        void *sbt[PF_BT_DEPTH];
        pf_print("use", sbt, pf_capture_exact(sbt));
        std::fflush(stderr);
        std::abort();
    }
    std::fprintf(stderr,
        "\n*** STM USE-AFTER-FREE ***\n"
        "  %s %p accessed via %s() by tid %d\n"
        "  freed by tid %d\n",
        h->type ? h->type : what, p, where, pf_tid(), h->death_tid);
    pf_print("free", h->death_bt, h->death_bt_n);
    pf_print("alloc", h->birth_bt, h->birth_bt_n);
    void *bt[PF_BT_DEPTH];
    pf_print("use", bt, pf_capture_exact(bt));
    std::fflush(stderr);
    std::abort();
}

}} // namespace Transactional::detail

#define KAME_PF_NEW_DELETE(what)                                              \
    static void *operator new(std::size_t sz)                                 \
        { return ::Transactional::detail::packet_forensics_new(sz, what); }   \
    static void operator delete(void *p) noexcept                             \
        { ::Transactional::detail::packet_forensics_delete(p); }              \
    static void operator delete(void *p, std::size_t) noexcept                \
        { ::Transactional::detail::packet_forensics_delete(p); }
#define KAME_PF_CK2(what, where)                                              \
    ::Transactional::detail::packet_forensics_check(this, what, where)
#define KAME_PF_CK(where) KAME_PF_CK2("Packet", where)

#elif defined KAME_STM_ALLOC_EXCLUSIVITY

// ---------------------------------------------------------------------------
// Layout-preserving arm.
//
// The header mode above adds 424 bytes to every object, which puts a ~40-byte
// Packet in a completely different pool size class -- so a clean soak under it
// says little about a fault that the record ties to allocator reuse.  This
// mode changes NOTHING about the allocation: it calls the same global
// operator new the default would have called, and keeps its bookkeeping in a
// side table.  It answers exactly one question, the one that separates the two
// readings of the poison evidence:
//
//     does the allocator ever hand out a block that is still live?
//
// The table is direct-mapped (hash of the address -> the address), not a real
// map.  A collision evicts a live entry, which costs a MISSED detection and
// can never manufacture one: a report requires the slot to still hold the very
// address being handed out, and the matching free clears that same slot.
// Objects here live for microseconds, so the window an entry must survive is
// tiny and eviction barely matters.
//
// What this arm cannot see is a stale READ -- there is no per-object stamp to
// check.  Use KAME_STM_PACKET_FORENSICS for that, and read its result knowing
// the size classes moved.
// ---------------------------------------------------------------------------

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace Transactional { namespace detail {

enum : std::size_t {
    PF_SHADOW_BITS  = 23,
    PF_SHADOW_SLOTS = std::size_t{1} << PF_SHADOW_BITS,
};

//! One slot per hash bucket.  `addr` is the live address (0 = free); the
//! rest is provenance for the report, written before `addr` is published so
//! a reader that matches `addr` sees a consistent record.
//! (§tier-trace) The allocator publishes which of its ~8 paths served the
//! most recent allocation on this thread, in a TLS word we read straight
//! after the call.  Weak, so a pool built without KAME_ALLOC_TIER_TRACE
//! still links and simply reports tier 0.
extern "C" __thread unsigned g_kame_alloc_tier __attribute__((weak));
inline unsigned pf_alloc_tier() noexcept {
    return ( &g_kame_alloc_tier) ? g_kame_alloc_tier : 0u;
}
inline const char *pf_tier_name(unsigned t) noexcept {
    switch(t) {
    case 1: return "new_redirected: TLS cell freelist pop";
    case 2: return "new_redirected_cold: cell pop";
    case 3: return "new_redirected_cold: word-cache ctz serve";
    case 4: return "cold_first_access";
    case 5: return "slow_allocate: scan_dll_freelist hit";
    case 6: return "slow_allocate: allocate_chunk_path";
    case 7: return "allocate_pooled: freelist_pop(0)";
    case 8: return "allocate_pooled: whole-word grab";
    case 9: return "allocate_pooled: bitmap scan";
    default: return "(untraced pool build)";
    }
}

struct PfSlot {
    std::atomic<std::uintptr_t> addr;
    const char *type;
    int tid;
    std::size_t size;
    unsigned tier;
};
inline PfSlot *pf_shadow() noexcept {
    static PfSlot *t = static_cast<PfSlot *>(
        std::calloc(PF_SHADOW_SLOTS, sizeof(PfSlot)));
    return t;
}
inline int pf_excl_tid() noexcept {
    static std::atomic<int> next{0};
    static thread_local int id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}
//! Per-type allocation / free tallies.  A type whose two tallies differ has
//! a free path that does not run through `excl_delete`, which is the ONE way
//! this arm can report a violation that is not real -- a slot can only still
//! hold an address if no `excl_delete` for it ran, so a bypassed free leaves
//! a stale record that the next allocation of that address trips over.
//! Printed at exit so the report can be read against them.
enum : int {PF_NTYPE = 4};
inline const char *pf_type_names(int i) noexcept {
    static const char *n[PF_NTYPE] =
        {"Packet", "PacketWrapper", "PacketList", "Payload"};
    return n[i];
}
inline std::atomic<std::size_t> *pf_tally(bool freed) noexcept {
    static std::atomic<std::size_t> a[PF_NTYPE], f[PF_NTYPE];
    return freed ? f : a;
}
inline int pf_type_index(const char *t) noexcept {
    for(int i = 0; i < PF_NTYPE; ++i)
        if(t == pf_type_names(i)) return i;     // literals are pooled per TU
    for(int i = 0; i < PF_NTYPE; ++i) {
        const char *a = t, *b = pf_type_names(i);
        while( *a && *a == *b) { ++a; ++b; }
        if( !*a && !*b) return i;
    }
    return -1;
}
inline void pf_excl_report_at_exit() noexcept {
    static bool once = (std::atexit([]{
        for(int i = 0; i < PF_NTYPE; ++i) {
            std::size_t a = pf_tally(false)[i].load(), f = pf_tally(true)[i].load();
            std::fprintf(stderr, "alloc-exclusivity: %-14s %zu alloc, %zu free%s\n",
                pf_type_names(i), a, f,
                (a == f) ? "" : "   <-- MISMATCH: a free path bypasses operator delete");
        }
    }), true);
    (void)once;
}
inline std::size_t pf_shadow_slot(std::uintptr_t a) noexcept {
    a >>= 4;                                    // pool blocks are 16-aligned
    a *= 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(a >> (64 - PF_SHADOW_BITS));
}

//! `KAME_PF_NO_TALLY` drops the per-type counters and the slot provenance.
//! They cost two atomic RMWs and three stores per allocation, which is not
//! nothing at ~550k allocations/s: the first configuration that reported a
//! violation had none of them, and adding them stopped it reporting.  Until
//! that is explained, the lean build has to stay reproducible on demand --
//! the difference between "the bookkeeping perturbed the race away" and
//! "the violation was an artefact of the leaner code" is the whole question.
inline void *excl_new(std::size_t sz, const char *type) {
#ifndef KAME_PF_NO_TALLY
    pf_excl_report_at_exit();
    if(int i = pf_type_index(type); i >= 0)
        pf_tally(false)[i].fetch_add(1, std::memory_order_relaxed);
#endif
    void *p = ::operator new(sz);               // exactly the default path
    unsigned tier = pf_alloc_tier();
    if(auto *t = pf_shadow()) {
        auto a = reinterpret_cast<std::uintptr_t>(p);
        PfSlot &slot = t[pf_shadow_slot(a)];
        // Acquire pairs with the release in excl_delete: the allocator's own
        // free->alloc edge already orders the two, but making it explicit
        // means a stale read here cannot be blamed on this table's ordering.
        if(slot.addr.load(std::memory_order_acquire) == a) {
            std::fprintf(stderr,
                "\n*** ALLOCATOR RETURNED A LIVE BLOCK ***\n"
                "  %p handed out for a %s (%zu bytes) on tid %d,\n"
                "  but our records still show it live as a %s (%zu bytes)\n"
                "  from tid %d -- no operator delete for it ever ran.\n"
                "  served the FIRST time by  tier %u: %s\n"
                "  served THIS time by       tier %u: %s\n"
                "  Read this against the per-type tallies below: they are\n"
                "  equal iff every free went through operator delete, which\n"
                "  is what makes the record trustworthy.\n",
                p, type, sz, pf_excl_tid(),
                slot.type ? slot.type : "?", slot.size, slot.tid,
                slot.tier, pf_tier_name(slot.tier),
                tier, pf_tier_name(tier));
            for(int i = 0; i < PF_NTYPE; ++i)
                std::fprintf(stderr, "    %-14s %zu alloc, %zu free\n",
                    pf_type_names(i), pf_tally(false)[i].load(),
                    pf_tally(true)[i].load());
            std::fflush(stderr);
            std::abort();
        }
#ifndef KAME_PF_NO_TALLY
        slot.type = type;
        slot.tid = pf_excl_tid();
        slot.size = sz;
        slot.tier = tier;
#endif
        slot.addr.store(a, std::memory_order_release);
    }
    return p;
}
//! \a type is the STATIC type whose operator delete ran, so the tally counts
//! every free regardless of whether the slot record survived eviction --
//! which is what makes "alloc == free" a real statement about coverage.
inline void excl_delete(void *p, const char *type) noexcept {
    if(p) {
#ifndef KAME_PF_NO_TALLY
        if(int i = pf_type_index(type); i >= 0)
            pf_tally(true)[i].fetch_add(1, std::memory_order_relaxed);
#else
        (void)type;
#endif
        if(auto *t = pf_shadow()) {
            auto a = reinterpret_cast<std::uintptr_t>(p);
            PfSlot &slot = t[pf_shadow_slot(a)];
            if(slot.addr.load(std::memory_order_relaxed) == a)
                slot.addr.store(0, std::memory_order_release);
        }
        ::operator delete(p);
    }
}

}} // namespace Transactional::detail

#define KAME_PF_NEW_DELETE(what)                                              \
    static void *operator new(std::size_t sz)                                 \
        { return ::Transactional::detail::excl_new(sz, what); }               \
    static void operator delete(void *p) noexcept                             \
        { ::Transactional::detail::excl_delete(p, what); }                    \
    static void operator delete(void *p, std::size_t) noexcept                \
        { ::Transactional::detail::excl_delete(p, what); }
#define KAME_PF_CK(where) ((void)0)
#define KAME_PF_CK2(what, where) ((void)0)

#else  // no forensics

#define KAME_PF_NEW_DELETE(what)
#define KAME_PF_CK(where) ((void)0)
#define KAME_PF_CK2(what, where) ((void)0)

#endif

#endif // PACKET_FORENSICS_H
