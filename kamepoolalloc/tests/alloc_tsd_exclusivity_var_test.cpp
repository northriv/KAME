// Exclusivity across thread teardown, variable-size (FS=false) blocks: the
// allocator must never hand out memory that is still live, nor write into it.
//
// alloc_tsd_exclusivity_test does this for 32 B blocks (FS=true, the
// buffering `push` arm).  Both thread-exit defects the pool has shipped sat
// in paths every size class shares -- a free from a thread past its own
// teardown (v1.1.0), a free racing its chunk owner's exit (v1.2.0) -- but the
// FS=false classes add their own state that outlives a thread: freelists per
// local id, the slot prefix and m_sizes, the borrow tier's re-aimed shortcut,
// §fl-scan's FL_UNTRACKED marks (left on an exiting thread's freelists for
// the next owner) and §walk's cursor.  Nothing asserted exclusivity there.
//
// Same shape as the 32 B test -- cross-thread frees through a shared handoff
// array, pointers stranded in a pthread_key TSD so they are freed after the
// C++ thread_local teardown, generations of threads under a workload that
// outlives them -- with sizes over every FS=false tier:
//
//   * borrow tier, ALIGN 64 and 256: 369..4352 B (slot prefix at p - 8);
//   * full-usable tier, ALIGN 1024: 4353..17408 B (m_sizes);
//   * ALIGN 4096: the page-sized requests (4096, 8192, 16384) and
//     17409..32768 B;
//
// and an allocate-only burst at every thread start, so buckets go dry
// (§fl-scan marks every freelist but the shortcut's) and the frees that
// follow land on marked heads -- also on chunks inherited from an exited
// thread, whose freelists its exit left marked.
//
// Two checks.  An exact-address shadow table, as in the 32 B test, catches a
// block handed out again at the same start while live.  And every block is
// painted on allocation -- its first word carries a nonce and its size, one
// word per 64 B (and its last word) a value derived from address, nonce and
// offset -- and verified before it is freed, which also catches a block that
// partly overlaps another live one (a different start), and any write the
// allocator makes into live memory (a stale freelist pushing onto it).
//
// Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.

#include "kame_pool.h"
#include <pthread.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t SHADOW_BITS = 20;
constexpr std::size_t SHADOW      = std::size_t{1} << SHADOW_BITS;
constexpr int         HANDOFF     = 4096;  // cross-thread free channel
constexpr int         GENERATIONS = 48;
constexpr int         THREADS     = 8;
constexpr int         ITERS       = 4000;
constexpr int         BURST       = 96;    // allocate-only at thread start
constexpr int         STRAND      = 64;    // pointers freed from the TSD dtor
constexpr std::size_t STRIDE      = 64;    // one painted word per 64 B

std::atomic<std::uintptr_t> *g_shadow;
std::atomic<void *>          g_handoff[HANDOFF];
std::atomic<int>             g_violations{0};
std::atomic<long>            g_blocks{0};

std::size_t shadow_slot(std::uintptr_t a) noexcept {
    // In uint64_t: on ILP32 a uintptr_t multiply would truncate (see the 32 B
    // test).
    std::uint64_t h = static_cast<std::uint64_t>(a) >> 4;
    h *= 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(h >> (64 - SHADOW_BITS));
}

std::uint64_t mix(std::uint64_t z) noexcept {     // splitmix64's finaliser
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

//! The size classes: every FS=false tier, the page-sized requests included.
std::size_t pick_size(unsigned &rng) noexcept {
    rng = rng * 1664525u + 1013904223u;
    unsigned r = (rng >> 8) % 100u;
    unsigned v = rng >> 4;
    if(r < 55) return 369 + v % (4352 - 369 + 1);        // borrow: ALIGN 64, 256
    if(r < 85) return 4353 + v % (17408 - 4353 + 1);     // full: ALIGN 1024
    if(r < 93) return std::size_t{4096} << (v % 3);      // ALIGN 4096, page-sized
    return 17409 + v % (32768 - 17409 + 1);              // ALIGN 4096, bucket 51
}

void fail(const char *what, void *p, std::size_t size, std::size_t off,
          std::uint64_t want, std::uint64_t got) {
    if(g_violations.fetch_add(1, std::memory_order_relaxed) == 0)
        std::fprintf(stderr,
            "FAIL: %s: block %p (%zu B), offset %zu: expected 0x%016llx, "
            "found 0x%016llx\n", what, p, size, off,
            static_cast<unsigned long long>(want),
            static_cast<unsigned long long>(got));
}

//! Word 0: nonce in the top 48 bits, size in the low 16 (sizes <= 32768).
void paint(void *p, std::size_t size, std::uint64_t nonce) noexcept {
    auto *w = static_cast<std::uint64_t *>(p);
    const std::uint64_t a = reinterpret_cast<std::uintptr_t>(p);
    w[0] = (nonce << 16) | size;
    for(std::size_t off = STRIDE; off + 8 <= size; off += STRIDE)
        w[off / 8] = mix(a ^ (nonce * 0x9E3779B97F4A7C15ull) ^ off);
    const std::size_t last = (size & ~std::size_t{7}) - 8;
    if(last % STRIDE != 0)
        w[last / 8] = mix(a ^ (nonce * 0x9E3779B97F4A7C15ull) ^ last);
}

void verify(void *p) noexcept {
    auto *w = static_cast<std::uint64_t *>(p);
    const std::uint64_t a = reinterpret_cast<std::uintptr_t>(p);
    const std::uint64_t head = w[0];
    const std::size_t size = static_cast<std::size_t>(head & 0xFFFFu);
    const std::uint64_t nonce = head >> 16;
    if(size < 369 || size > 32768) {
        // The size cannot be wrong unless word 0 was overwritten.
        fail("first word overwritten", p, size, 0, 0, head);
        return;
    }
    for(std::size_t off = STRIDE; off + 8 <= size; off += STRIDE) {
        const std::uint64_t want = mix(a ^ (nonce * 0x9E3779B97F4A7C15ull) ^ off);
        if(w[off / 8] != want) { fail("live block overwritten", p, size, off, want, w[off / 8]); return; }
    }
    const std::size_t last = (size & ~std::size_t{7}) - 8;
    if(last % STRIDE != 0) {
        const std::uint64_t want = mix(a ^ (nonce * 0x9E3779B97F4A7C15ull) ^ last);
        if(w[last / 8] != want) fail("live block overwritten", p, size, last, want, w[last / 8]);
    }
}

void *tracked_alloc(unsigned &rng, std::uint64_t nonce) {
    const std::size_t size = pick_size(rng);
    void *p = kame_pool_malloc(size);
    if( !p) return nullptr;
    auto a = reinterpret_cast<std::uintptr_t>(p);
    auto &slot = g_shadow[shadow_slot(a)];
    if(slot.load(std::memory_order_acquire) == a)
        fail("handed out while live (same start)", p, size, 0, 0, 0);
    else
        slot.store(a, std::memory_order_release);
    paint(p, size, nonce);
    g_blocks.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void tracked_free(void *p) {
    if( !p) return;
    verify(p);
    auto a = reinterpret_cast<std::uintptr_t>(p);
    auto &slot = g_shadow[shadow_slot(a)];
    if(slot.load(std::memory_order_relaxed) == a)
        slot.store(0, std::memory_order_release);
    kame_pool_free(p);
}

// Pointers stranded here are freed by the pthread_key destructor, i.e.
// after every C++ thread_local destructor of the exiting thread.
pthread_key_t g_key;
void tsd_destructor(void *arg) {
    auto *v = static_cast<std::vector<void *> *>(arg);
    for(void *p : *v) tracked_free(p);
    delete v;
}

void worker(int seed) {
    auto *stranded = new std::vector<void *>;
    stranded->reserve(STRAND);
    pthread_setspecific(g_key, stranded);

    unsigned rng = static_cast<unsigned>(seed) * 2654435761u + 1u;
    std::uint64_t nonce = (static_cast<std::uint64_t>(seed) << 24) + 1;
    auto next_nonce = [&]() { return ++nonce & 0xFFFFFFFFFFFFull; };

    // Allocate-only: misses with nothing to find, so buckets go dry and
    // freelists get marked.  Half of it goes to other threads to free.
    std::vector<void *> burst;
    burst.reserve(BURST);
    for(int i = 0; i < BURST; ++i)
        if(void *p = tracked_alloc(rng, next_nonce())) burst.push_back(p);
    for(std::size_t i = 0; i < burst.size(); ++i) {
        void *p = burst[i];
        if(i & 1) {
            void *expect = nullptr;
            int idx = static_cast<int>(mix(reinterpret_cast<std::uintptr_t>(p)) % HANDOFF);
            if(g_handoff[idx].compare_exchange_strong(expect, p,
                    std::memory_order_release, std::memory_order_relaxed))
                continue;
        }
        tracked_free(p);              // an owner free onto a marked head
    }

    for(int i = 0; i < ITERS; ++i) {
        rng = rng * 1664525u + 1013904223u;
        int idx = static_cast<int>((rng >> 8) % HANDOFF);

        // Take someone else's block -- possibly from a thread that has exited,
        // whose chunk this thread may since have taken over -- and free it.
        if(void *taken = g_handoff[idx].exchange(nullptr, std::memory_order_acquire))
            tracked_free(taken);

        void *p = tracked_alloc(rng, next_nonce());
        if( !p) continue;

        if(static_cast<int>(stranded->size()) < STRAND && (rng & 0x3f) == 0) {
            stranded->push_back(p);          // freed at teardown, from the TSD
            continue;
        }
        void *expect = nullptr;
        if((rng & 3) == 0 || !g_handoff[idx].compare_exchange_strong(expect, p,
                std::memory_order_release, std::memory_order_relaxed))
            tracked_free(p);                 // an owner free
    }
}

} // namespace

int main() {
    g_shadow = static_cast<std::atomic<std::uintptr_t> *>(
        std::calloc(SHADOW, sizeof(std::atomic<std::uintptr_t>)));
    if( !g_shadow) { std::fprintf(stderr, "FAIL: shadow calloc\n"); return 1; }
    if(pthread_key_create( &g_key, &tsd_destructor) != 0) {
        std::fprintf(stderr, "FAIL: pthread_key_create\n");
        return 1;
    }

    for(int g = 0; g < GENERATIONS && !g_violations.load(); ++g) {
        std::vector<std::thread> ts;
        ts.reserve(THREADS);
        for(int t = 0; t < THREADS; ++t)
            ts.emplace_back(worker, g * THREADS + t);
        for(auto &t : ts) t.join();
    }

    for(int i = 0; i < HANDOFF; ++i)
        if(void *p = g_handoff[i].exchange(nullptr)) tracked_free(p);

    int v = g_violations.load();
    if(v) {
        std::fprintf(stderr, "FAILED: %d exclusivity violation(s) in %ld blocks\n",
                     v, g_blocks.load());
        return 1;
    }
    std::printf("succeeded (%ld blocks)\n", g_blocks.load());
    return 0;
}
