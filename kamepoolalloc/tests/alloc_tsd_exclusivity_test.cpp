// Exclusivity across thread teardown: the allocator must never hand out a
// block that is still live.
//
// The defect this guards (see design/BATCH_AFTER_DESTRUCTOR.md): glibc runs
// __call_tls_dtors() -- C++ thread_local destructors, hence
// ~CrossDeallocBatch -- BEFORE __nptl_deallocate_tsd(), the pthread_key
// destructors.  KAME registers one of the latter
// (Transactional::detail::pthread_destroy, kamestm/threadlocal.cpp) which
// frees STM objects, so those frees reached a DESTROYED cross-dealloc batch
// and resurrected it.  A slot then went back to the bitmap after its owner
// had re-issued it, and the whole-word grab / bitmap scan handed the same
// block to a second live user.  Downstream that showed up as garbage inside
// a still-referenced STM Packet, four levels away from the cause.
//
// alloc_thread_exit_free_test already models the same teardown ORDERING, but
// asserts against leaks; nothing asserted exclusivity, which is why this went
// unnoticed.  The shape reproduced here is the one the fault needs:
//
//   * cross-thread frees, so the cross-dealloc batch is actually used
//     (an owner-side free takes the freelist and never batches),
//   * small blocks -- 32 B takes the buffering `push` arm, ALIGN <= 48,
//     not `push_direct`,
//   * frees stranded in a pthread_key TSD so they run after the C++
//     thread_local teardown,
//   * many thread generations under a workload that outlives them.
//
// The check is a direct-mapped address -> address table.  A collision only
// EVICTS a record, which loses a detection and cannot manufacture one: a
// report requires the slot to still hold the very address being handed out,
// and the matching free clears that same slot.
//
// Pool-only (kame_pool_* C API); built when USE_KAME_ALLOCATOR is ON.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t BLOCK       = 32;    // ALIGN <= 48 -> the `push` arm
constexpr std::size_t SHADOW_BITS = 20;
constexpr std::size_t SHADOW      = std::size_t{1} << SHADOW_BITS;
constexpr int         HANDOFF     = 4096;  // cross-thread free channel
constexpr int         GENERATIONS = 24;
constexpr int         THREADS     = 8;
constexpr int         ITERS       = 20000;
constexpr int         STRAND      = 64;    // pointers freed from the TSD dtor

std::atomic<std::uintptr_t> *g_shadow;
std::atomic<void *>          g_handoff[HANDOFF];
std::atomic<int>             g_violations{0};
std::atomic<bool>            g_stop{false};

std::size_t shadow_slot(std::uintptr_t a) noexcept {
    // Mix in uint64_t, not in uintptr_t: on ILP32 the multiply would truncate and
    // `>> (64 - SHADOW_BITS)` would shift a 32-bit value by 44 — undefined, and in
    // practice every address collapsing onto slot 0, i.e. a test that reports
    // nothing.  i486 is one of the platforms the original fault reproduced on.
    std::uint64_t h = static_cast<std::uint64_t>(a) >> 4;
    h *= 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(h >> (64 - SHADOW_BITS));
}

void *tracked_alloc() {
    void *p = kame_pool_malloc(BLOCK);
    if( !p) return nullptr;
    auto a = reinterpret_cast<std::uintptr_t>(p);
    auto &slot = g_shadow[shadow_slot(a)];
    if(slot.load(std::memory_order_acquire) == a) {
        if(g_violations.fetch_add(1, std::memory_order_relaxed) == 0)
            std::fprintf(stderr,
                "FAIL: allocator returned %p while our records still show it "
                "live -- no free for it had been seen\n", p);
        return p;
    }
    slot.store(a, std::memory_order_release);
    return p;
}

void tracked_free(void *p) {
    if( !p) return;
    auto a = reinterpret_cast<std::uintptr_t>(p);
    auto &slot = g_shadow[shadow_slot(a)];
    if(slot.load(std::memory_order_relaxed) == a)
        slot.store(0, std::memory_order_release);
    kame_pool_free(p);
}

// Pointers stranded here are freed by the pthread_key destructor, i.e.
// during __nptl_deallocate_tsd -- after every C++ thread_local destructor.
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
    for(int i = 0; i < ITERS && !g_stop.load(std::memory_order_relaxed); ++i) {
        rng = rng * 1664525u + 1013904223u;
        int idx = static_cast<int>((rng >> 8) % HANDOFF);

        // Take someone else's block and free it: a CROSS-THREAD free, which
        // is what routes through the batch at all.
        if(void *taken = g_handoff[idx].exchange(nullptr,
                                                 std::memory_order_acquire))
            tracked_free(taken);

        void *p = tracked_alloc();
        if( !p) continue;

        if(static_cast<int>(stranded->size()) < STRAND && (rng & 0x3f) == 0) {
            stranded->push_back(p);          // freed at teardown, from the TSD
            continue;
        }
        // Offer it for another thread to free; if the slot is taken, keep the
        // free local.
        void *expect = nullptr;
        if( !g_handoff[idx].compare_exchange_strong(expect, p,
                std::memory_order_release, std::memory_order_relaxed))
            tracked_free(p);
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
    g_stop.store(true);

    for(int i = 0; i < HANDOFF; ++i)
        if(void *p = g_handoff[i].exchange(nullptr)) tracked_free(p);

    int v = g_violations.load();
    if(v) {
        std::fprintf(stderr, "FAILED: %d exclusivity violation(s)\n", v);
        return 1;
    }
    std::printf("succeeded\n");
    return 0;
}
