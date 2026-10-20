// (§ded-fast) The dedicated tier's fast paths -- 32 KiB < size ≤ 4 MiB − K_MAX
// served from, and freed into, the thread's L1 recycle cache off KameTlsPage.
//
// The fast pop takes only a block of DEDICATED origin and skips the re-stamp
// of its header and back_offset bytes; the fast free puts the chunk on the
// L1 itself; everything else falls to the general path.  These scenarios
// drive each boundary of that split:
//
//   sizes      every unit count 1..16 at both edges of its size class, many
//              times over, so the first call goes the general way and the
//              rest the fast one; the usable size must be the block's actual
//              size (index 12 holds both 8- and 9-unit chunks, so an 8-unit
//              request may get a 9-unit block).
//   overflow   more blocks in flight than the L1 has slots (32): the frees
//              past it go to deallocate_cold → the global L2.
//   buckets    16 KiB slots freed in bulk release whole bucket chunks into
//              the same cache (§34); a later dedicated request must still
//              get a block of the right size, converted by the general path.
//   calloc     a recycled block is handed back unzeroed by the cache, so
//              calloc must still zero it.
//   realloc    across the tier and out of it, contents kept.
//   aligned    alignments up to 256 KiB come from a dedicated chunk; the
//              block then recycles through the same fast paths.
//   threads    blocks allocated on one thread, freed on another (into ITS
//              L1), which then exits (drain to L2), reused by the first.
//   late       dedicated blocks allocated and freed from a thread_local
//              destructor, in both orders a program can get: registered
//              before the thread's first allocation (so destroyed after all
//              of the allocator's per-thread state: the thread is torn down,
//              requests go to the system heap) and registered after it
//              (destroyed before the L1 drain: the fast paths still serve
//              it).  A TU's thread_local destructors are registered together
//              on its first access, so allocator.cpp's L1 drain and its
//              torn-down flag go out back to back and no user destructor runs
//              between them; l1_pop_fit refuses to re-arm a drained L1 all
//              the same -- re-arming it would now also re-arm the page's copy
//              the fast free trusts.  Either order: after kame_pool_trim()
//              every dedicated byte the scenarios claimed must be gone
//              (counted from what was live before them -- stdout's buffer
//              can be a dedicated block; see g_base_bytes).
//
// Pool-only (kame_pool_* C API).  Built static (allocator.cpp compiled in,
// as kame.exe / kame.app ship it) and dynamic.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); \
    std::printf("\n"); } } while(0)

static const std::size_t UNIT  = (std::size_t)256 * 1024;   // ALLOC_MIN_CHUNK_SIZE
static const std::size_t K_MAX = 4096;                      // ALLOC_CHUNK_K_MAX
static const std::size_t BUCKETED_MAX = 32768;              // ALLOC_MAX_BUCKETED_SIZE

static kame_pool_stats_t stats() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s;
}

// Units a request of `size` occupies (allocate_dedicated_chunk's rounding).
static std::size_t units_for(std::size_t size) { return (size + K_MAX + UNIT - 1) / UNIT; }

// Fill the first and last 64 bytes, and one in the middle, with a tag.
static void stamp(void *p, std::size_t n, unsigned char tag) {
    unsigned char *c = static_cast<unsigned char *>(p);
    std::memset(c, tag, n < 64 ? n : 64);
    std::memset(c + n - (n < 64 ? n : 64), tag, n < 64 ? n : 64);
    c[n / 2] = tag;
}
static bool stamped(const void *p, std::size_t n, unsigned char tag) {
    const unsigned char *c = static_cast<const unsigned char *>(p);
    const std::size_t e = n < 64 ? n : 64;
    for(std::size_t i = 0; i < e; ++i)
        if(c[i] != tag || c[n - e + i] != tag) return false;
    return c[n / 2] == tag;
}

// A block of this tier must report a usable size of whole units minus the
// metadata page, at least the request and at most one unit more (index band).
static void check_block(void *p, std::size_t size, const char *what) {
    CHECK(p != nullptr, "%s: malloc(%zu) returned null", what, size);
    if( !p) return;
    CHECK(((std::uintptr_t)p & (UNIT - 1)) == 0,
          "%s: %zu-byte block %p is not on a 256 KiB boundary", what, size, p);
    const std::size_t us = kame_pool_malloc_usable_size(p);
    const std::size_t u = units_for(size);
    CHECK(us >= size && (us + K_MAX) % UNIT == 0
          && (us + K_MAX) / UNIT >= u && (us + K_MAX) / UNIT <= u + 1,
          "%s: malloc(%zu) usable %zu (expected %zu or %zu units - K_MAX)",
          what, size, us, u, u + 1);
}

static void scenario_sizes() {
    std::printf("[sizes] unit counts 1..16 at both edges, repeated\n");
    for(int round = 0; round < 50; ++round) {
        for(std::size_t u = 1; u <= 16; ++u) {
            const std::size_t lo = (u == 1) ? BUCKETED_MAX + 1 : (u - 1) * UNIT - K_MAX + 1;
            const std::size_t hi = u * UNIT - K_MAX;
            for(std::size_t size : {lo, hi}) {
                void *p = kame_pool_malloc(size);
                check_block(p, size, "sizes");
                if( !p) continue;
                const std::size_t us = kame_pool_malloc_usable_size(p);
                stamp(p, us, (unsigned char)(u + round));
                CHECK(stamped(p, us, (unsigned char)(u + round)), "sizes: stamp lost");
                kame_pool_free(p);
            }
        }
    }
    // An 8-unit request after a 9-unit block was freed: the shared index may
    // hand the larger block back; its usable size must say so.
    void *big = kame_pool_malloc(9 * UNIT - K_MAX);
    check_block(big, 9 * UNIT - K_MAX, "band");
    kame_pool_free(big);
    void *small = kame_pool_malloc(8 * UNIT - K_MAX);
    check_block(small, 8 * UNIT - K_MAX, "band");
    if(small) {
        const std::size_t us = kame_pool_malloc_usable_size(small);
        stamp(small, us, 0x5A);
        CHECK(stamped(small, us, 0x5A), "band: stamp lost over the usable size");
    }
    kame_pool_free(small);
}

static void scenario_overflow() {
    std::printf("[overflow] 48 blocks in flight (> 32 L1 slots), 3 rounds\n");
    const int N = 48;
    for(int round = 0; round < 3; ++round) {
        std::vector<void *> v;
        for(int i = 0; i < N; ++i) {
            void *p = kame_pool_malloc(64 * 1024);
            check_block(p, 64 * 1024, "overflow");
            if(p) { stamp(p, 64 * 1024, (unsigned char)i); v.push_back(p); }
        }
        for(std::size_t i = 0; i < v.size(); ++i)
            for(std::size_t j = i + 1; j < v.size(); ++j)
                CHECK(v[i] != v[j], "overflow: block %p handed out twice", v[i]);
        for(std::size_t i = 0; i < v.size(); ++i)
            CHECK(stamped(v[i], 64 * 1024, (unsigned char)i), "overflow: block %zu overwritten", i);
        if(round & 1) for(void *p : v) kame_pool_free(p);
        else for(std::size_t i = v.size(); i-- > 0;) kame_pool_free(v[i]);
    }
}

static void scenario_buckets() {
    std::printf("[buckets] 16 KiB chunks released into the cache, then dedicated requests\n");
    for(int round = 0; round < 4; ++round) {
        std::vector<void *> v;
        for(int i = 0; i < 3000; ++i)                     // ~47 MiB of 16 KiB slots
            if(void *p = kame_pool_malloc(16 * 1024)) v.push_back(p);
        for(void *p : v) kame_pool_free(p);               // past the 4 MiB list cap → chunks released
        for(std::size_t size : {(std::size_t)40 * 1024, UNIT - K_MAX, 2 * UNIT - K_MAX,
                                4 * UNIT - K_MAX, (std::size_t)100 * 1024}) {
            for(int k = 0; k < 40; ++k) {
                void *p = kame_pool_malloc(size);
                check_block(p, size, "buckets");
                if( !p) continue;
                const std::size_t us = kame_pool_malloc_usable_size(p);
                stamp(p, us, 0xC3);
                CHECK(stamped(p, us, 0xC3), "buckets: stamp lost");
                kame_pool_free(p);
            }
        }
    }
}

static void scenario_calloc_realloc() {
    std::printf("[calloc/realloc] recycled blocks zeroed by calloc; realloc across the tier\n");
    for(int round = 0; round < 20; ++round) {
        const std::size_t n = (std::size_t)(48 + round * 37) * 1024;
        void *p = kame_pool_malloc(n);
        if(p) { std::memset(p, 0xEE, n); kame_pool_free(p); }
        unsigned char *z = static_cast<unsigned char *>(kame_pool_calloc(1, n));
        CHECK(z != nullptr, "calloc(%zu) returned null", n);
        if( !z) continue;
        bool zero = true;
        for(std::size_t i = 0; i < n; i += 61) zero = zero && z[i] == 0;
        zero = zero && z[n - 1] == 0;
        CHECK(zero, "calloc(%zu): recycled block not zeroed", n);
        kame_pool_free(z);
    }
    unsigned char *p = static_cast<unsigned char *>(kame_pool_malloc(40 * 1024));
    for(std::size_t i = 0; i < 40 * 1024; ++i) p[i] = (unsigned char)(i * 7);
    for(std::size_t n : {(std::size_t)300 * 1024, (std::size_t)3 * 1024 * 1024,
                         (std::size_t)70 * 1024, (std::size_t)20 * 1024,
                         (std::size_t)900 * 1024}) {
        p = static_cast<unsigned char *>(kame_pool_realloc(p, n));
        CHECK(p != nullptr, "realloc to %zu returned null", n);
        if( !p) return;
        const std::size_t keep = n < 20 * 1024 ? n : 20 * 1024;
        bool same = true;
        for(std::size_t i = 0; i < keep; ++i) same = same && p[i] == (unsigned char)(i * 7);
        CHECK(same, "realloc to %zu lost the contents", n);
    }
    kame_pool_free(p);
}

static void scenario_aligned() {
    std::printf("[aligned] 64..256 KiB alignment from dedicated chunks, recycled\n");
    for(int round = 0; round < 30; ++round) {
        for(std::size_t al : {(std::size_t)64 * 1024, (std::size_t)128 * 1024, UNIT}) {
            const std::size_t n = (std::size_t)(100 + round * 13) * 1024;
            void *p = kame_pool_aligned_alloc(al, n);
            CHECK(p != nullptr && ((std::uintptr_t)p & (al - 1)) == 0,
                  "aligned_alloc(%zu, %zu) = %p", al, n, p);
            if( !p) continue;
            stamp(p, n, 0x77);
            kame_pool_free(p);
            void *q = kame_pool_malloc(n);              // the same block, by the fast pop
            check_block(q, n, "aligned");
            kame_pool_free(q);
        }
    }
}

static void scenario_threads() {
    std::printf("[threads] allocate on A, free on B (B's L1), B exits, A reuses\n");
    for(int round = 0; round < 30; ++round) {
        std::vector<void *> v;
        for(int i = 0; i < 40; ++i) {
            const std::size_t n = (std::size_t)(33 + (i * 97) % 3000) * 1024;
            void *p = kame_pool_malloc(n);
            check_block(p, n, "threads");
            if(p) { stamp(p, n, (unsigned char)i); v.push_back(p); }
        }
        std::thread([&v] {
            for(void *p : v) kame_pool_free(p);
            // B also churns a little on its own, so its L1 holds both.
            for(int k = 0; k < 20; ++k) kame_pool_free(kame_pool_malloc(200 * 1024));
        }).join();
        for(int i = 0; i < 40; ++i) {
            void *p = kame_pool_malloc(64 * 1024);
            check_block(p, 64 * 1024, "threads-reuse");
            kame_pool_free(p);
        }
    }
}

// Allocates and frees dedicated blocks when destroyed.  Its destructor is
// registered where a thread first touches it: before the thread's first
// allocation in the odd rounds, after it in the even ones (see the header).
struct DtorUser {
    bool armed = false;
    ~DtorUser() {
        if( !armed) return;
        for(int i = 0; i < 8; ++i) {
            void *p = kame_pool_malloc(64 * 1024 + (std::size_t)i * 4096);
            if(p) std::memset(p, 0x3C, 64);
            kame_pool_free(p);
        }
    }
};
static thread_local DtorUser t_user;

// Dedicated bytes live before the scenarios: stdout's buffer is st_blksize
// bytes, allocated on the first printf, and macOS's /dev/null reports 64 KiB,
// so in the dynamic build (malloc goes to the pool) a run with stdout on
// /dev/null holds one dedicated block to the end.  The checks after
// kame_pool_trim() count from here.
static std::size_t g_base_bytes = 0;

static void scenario_late() {
    std::printf("[late] dedicated alloc/free in thread_local dtors, both orders\n");
    for(int round = 0; round < 60; ++round) {
        const bool early = (round & 1) != 0;
        std::thread([early] {
            if(early) t_user.armed = true;                  // before any allocation
            kame_pool_free(kame_pool_malloc(100));
            for(int k = 0; k < 4; ++k)                      // arm the L1
                kame_pool_free(kame_pool_malloc(64 * 1024));
            if( !early) t_user.armed = true;                // after the allocator's state
        }).join();
    }
    kame_pool_trim();
    const kame_pool_stats_t s = stats();
    CHECK(s.dedicated_chunk_bytes <= g_base_bytes,
          "late: %zu dedicated bytes still claimed after trim, %zu live before "
          "(a block stranded in an L1 no drain will reach)",
          s.dedicated_chunk_bytes, g_base_bytes);
}

int main() {
    std::printf("alloc_dedicated_fast_test\n");
    std::fflush(stdout);
    kame_pool_trim();
    g_base_bytes = stats().dedicated_chunk_bytes;

    scenario_sizes();
    scenario_overflow();
    scenario_buckets();
    scenario_calloc_realloc();
    scenario_aligned();
    scenario_threads();

    kame_pool_trim();
    const kame_pool_stats_t s0 = stats();
    CHECK(s0.dedicated_chunk_bytes <= g_base_bytes,
          "%zu dedicated bytes still claimed after trim, %zu live before",
          s0.dedicated_chunk_bytes, g_base_bytes);

    scenario_late();

    if(g_fails) {
        std::printf("alloc_dedicated_fast_test: %d FAILURE(S)\n", g_fails);
        return 1;
    }
    std::printf("alloc_dedicated_fast_test: OK\n");
    return 0;
}
