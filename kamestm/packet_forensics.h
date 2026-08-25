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
inline void pf_report_at_exit() noexcept {
    static bool once = (std::atexit([]{
        std::fprintf(stderr, "packet-forensics: %zu Packet alloc, %zu free\n",
            pf_allocs().load(), pf_frees().load());
    }), true);
    (void)once;
}

inline void *packet_forensics_new(std::size_t sz) {
    pf_report_at_exit();
    pf_allocs().fetch_add(1, std::memory_order_relaxed);
    void *raw = pf_raw_alloc(sizeof(PacketForensicsHeader) + sz);
    if( !raw) throw std::bad_alloc();
    auto *h = static_cast<PacketForensicsHeader *>(raw);
    h->magic = PF_MAGIC_ALIVE;
    h->size = sz;
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
            "\n*** PACKET DOUBLE FREE at %p (first freed by tid %d) ***\n",
            p, h->death_tid);
        pf_print("first free", h->death_bt, h->death_bt_n);
        pf_print("second free", nullptr, 0);
        std::abort();
    }
    pf_frees().fetch_add(1, std::memory_order_relaxed);
    h->magic = PF_MAGIC_DEAD;
    h->death_tid = pf_tid();
    h->death_bt_n = pf_capture(h->death_bt);
    // Poison the body so a stale read of any member is obviously wrong even
    // if the magic check is bypassed.
    std::memset(p, 0xBD, h->size);

    PfRing &r = pf_ring();
    if( !r.slot) { pf_raw_free(h); return; }   // out of memory: no quarantine
    std::size_t i = r.pos++ % PF_QUARANTINE;
    void *old = r.slot[i];
    r.slot[i] = h;
    if(old) pf_raw_free(old);
}

//! \return silently when \a p is a live Packet; aborts with both stacks
//! otherwise.  \a where names the accessor for the report.
inline void packet_forensics_check(const void *p, const char *where) noexcept {
    if( !p) return;
    auto *h = static_cast<const PacketForensicsHeader *>(p) - 1;
    if(h->magic == PF_MAGIC_ALIVE) return;
    std::fprintf(stderr,
        "\n*** PACKET USE-AFTER-FREE ***\n"
        "  Packet %p accessed via %s by tid %d\n"
        "  header magic = 0x%016llx (expected ALIVE 0x%016llx)\n"
        "  freed by tid %d\n",
        p, where, pf_tid(),
        (unsigned long long)h->magic, (unsigned long long)PF_MAGIC_ALIVE,
        (h->magic == PF_MAGIC_DEAD) ? h->death_tid : -1);
    if(h->magic == PF_MAGIC_DEAD) {
        pf_print("free", h->death_bt, h->death_bt_n);
        pf_print("alloc", h->birth_bt, h->birth_bt_n);
    }
    void *bt[PF_BT_DEPTH];
    pf_print("use", bt, pf_capture_exact(bt));
    std::fflush(stderr);
    std::abort();
}

}} // namespace Transactional::detail

#define KAME_PF_NEW_DELETE                                                    \
    static void *operator new(std::size_t sz)                                 \
        { return ::Transactional::detail::packet_forensics_new(sz); }         \
    static void operator delete(void *p) noexcept                             \
        { ::Transactional::detail::packet_forensics_delete(p); }              \
    static void operator delete(void *p, std::size_t) noexcept                \
        { ::Transactional::detail::packet_forensics_delete(p); }
#define KAME_PF_CK(where)                                                     \
    ::Transactional::detail::packet_forensics_check(this, where)

#else  // !KAME_STM_PACKET_FORENSICS

#define KAME_PF_NEW_DELETE
#define KAME_PF_CK(where) ((void)0)

#endif

#endif // PACKET_FORENSICS_H
