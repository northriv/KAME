// (§29 watermark / §75 prewarm) Regression and RSS test for FS=true lazy
// watermark freelist pre-fill:
//   1. Progressive watermark expansion across all 17 FS=true buckets (16..368 B)
//      past one full chunk into a second chunk, checking byte-for-byte payload
//      exclusivity and post-trim release.
//   2. Partial-watermark cross-thread frees + orphan group adoption: a producer
//      exits while `m_prefill_word < m_count` with live cross-thread blocks; a
//      new owner adopts the group and expands `m_prefill_word` to `m_count`
//      without trampling any live block.
//   3. Multi-thread sparse-bucket RSS ceiling: 32 live threads touching 1 block
//      in each of the 17 FS=true buckets must fault ~1 OS page per chunk
//      (~2.2 MiB Linux / ~9.6 MiB macOS) instead of full 256..512 KiB chunks
//      (~192 MiB before §29 watermark).
//   4. `kame_pool_prewarm()` full-capacity guarantee: holding > 128 blocks
//      prewarms the full working set with zero OS mapping calls inside a
//      subsequent `kame_pool_set_realtime_thread(1)` critical section.

#include "kame_pool.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#  include <mach/mach.h>
#elif defined(__linux__)
#  include <unistd.h>
#endif

namespace {

constexpr std::size_t kFsTrueSizes[17] = {
    16, 32, 48, 64, 80, 112, 144, 176, 208, 240,
    272, 288, 304, 320, 336, 352, 368
};

int g_failures = 0;

#define CHECK(cond, fmt, ...)                                                  \
    do {                                                                       \
        if(!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL [%s:%d]: " fmt "\n", __FILE__,          \
                         __LINE__, ##__VA_ARGS__);                             \
            ++g_failures;                                                      \
        }                                                                      \
    } while(0)

kame_pool_stats_t get_stats() {
    kame_pool_stats_t st;
    std::memset(&st, 0, sizeof(st));
    st.version = KAME_POOL_STATS_VERSION;
    kame_pool_get_stats(&st);
    return st;
}

std::size_t current_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                 reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        return static_cast<std::size_t>(info.resident_size);
    return 0;
#elif defined(__linux__)
    std::FILE *f = std::fopen("/proc/self/statm", "r");
    if(!f) return 0;
    unsigned long total_pages = 0, rss_pages = 0;
    int got = std::fscanf(f, "%lu %lu", &total_pages, &rss_pages);
    std::fclose(f);
    if(got != 2) return 0;
    long ps = ::sysconf(_SC_PAGESIZE);
    return static_cast<std::size_t>(rss_pages) *
           static_cast<std::size_t>(ps > 0 ? ps : 4096);
#else
    return 0;
#endif
}

unsigned char pattern_byte(std::size_t bucket_idx, std::size_t item_idx,
                           std::size_t off) {
    return static_cast<unsigned char>(
        ((bucket_idx * 37u) ^ (item_idx * 131u) ^ (off * 17u) ^ 0x5Au) & 0xFFu);
}

void stamp_block(void *p, std::size_t sz, std::size_t bidx, std::size_t idx) {
    auto *b = static_cast<unsigned char *>(p);
    for(std::size_t off = 0; off < sz; ++off)
        b[off] = pattern_byte(bidx, idx, off);
}

bool verify_block(const void *p, std::size_t sz, std::size_t bidx,
                  std::size_t idx) {
    const auto *b = static_cast<const unsigned char *>(p);
    for(std::size_t off = 0; off < sz; ++off) {
        if(b[off] != pattern_byte(bidx, idx, off))
            return false;
    }
    return true;
}

// 1. Expand every FS=true bucket across > 1 full chunk, check integrity, trim.
void scenario_expand_all_buckets() {
    std::printf("[1/4] expand watermark across all 17 FS=true buckets\n");
    kame_pool_trim();
    const std::size_t u0 = get_stats().units_live;

    std::thread([&] {
        std::vector<void *> held[17];
        for(std::size_t b = 0; b < 17; ++b) {
            const std::size_t sz = kFsTrueSizes[b];
            // > 512 KiB per bucket guarantees > 1 full chunk at every FS=true size.
            const std::size_t count = (640u * 1024u) / sz;
            held[b].resize(count, nullptr);
            for(std::size_t i = 0; i < count; ++i) {
                void *p = kame_pool_malloc(sz);
                if(!p) {
                    CHECK(false, "malloc(%zu) returned null at i=%zu", sz, i);
                    break;
                }
                stamp_block(p, sz, b, i);
                held[b][i] = p;
            }
        }
        for(std::size_t b = 0; b < 17; ++b) {
            const std::size_t sz = kFsTrueSizes[b];
            std::size_t corrupted = 0;
            for(std::size_t i = 0; i < held[b].size(); ++i) {
                if(held[b][i] && !verify_block(held[b][i], sz, b, i))
                    ++corrupted;
                kame_pool_free(held[b][i]);
            }
            CHECK(corrupted == 0, "bucket size %zu had %zu corrupted blocks",
                  sz, corrupted);
            held[b].clear();
            held[b].shrink_to_fit();
        }
        kame_pool_trim();
        const std::size_t u_live = get_stats().units_live;
        // Live thread retains at most LEAVE_VACANT_CHUNKS_PER_THREAD (2) chunks
        // per touched template (~48..52 units across 17 FS=true templates).
        CHECK(u_live <= u0 + 60u,
              "live-thread units_live after trim too high: before=%zu live=%zu",
              u0, u_live);
    }).join();

    kame_pool_trim();
    const std::size_t u1 = get_stats().units_live;
    CHECK(u1 <= u0 + 4u,
          "post-exit units_live after trim leaked units: before=%zu after=%zu",
          u0, u1);
}

// 2. Partial watermark + cross-thread free + orphan adoption + full expansion.
void scenario_partial_orphan_adopt() {
    std::printf("[2/4] partial watermark cross-thread free + orphan adoption\n");
    for(int round = 0; round < 12; ++round) {
        constexpr std::size_t kSz = 64;
        constexpr std::size_t kPartial = 96;   // <= 1 watermark page on LP64
        std::vector<void *> live_cross(kPartial, nullptr);

        // Producer A allocates `kPartial` blocks (leaving m_prefill_word < m_count),
        // frees odd indices on its own list, and exits while even indices stay live.
        std::thread([&] {
            for(std::size_t i = 0; i < kPartial; ++i) {
                void *p = kame_pool_malloc(kSz);
                CHECK(p != nullptr, "partial malloc failed");
                if(p) stamp_block(p, kSz, static_cast<std::size_t>(round), i);
                if(i & 1u)
                    kame_pool_free(p);
                else
                    live_cross[i] = p;
            }
        }).join();

        // Consumer B frees half of the surviving blocks cross-thread (reviving
        // A's partially-watermarked chunk onto ROOM / FULL).
        std::thread([&] {
            for(std::size_t i = 0; i < kPartial; i += 4) {
                if(void *p = live_cross[i]) {
                    CHECK(verify_block(p, kSz, static_cast<std::size_t>(round), i),
                          "cross-free block %zu corrupted before free", i);
                    kame_pool_free(p);
                    live_cross[i] = nullptr;
                }
            }
            kame_pool_trim();
        }).join();

        // Adopter C takes over A's orphaned group and allocates 5,000 blocks,
        // forcing the adopted chunk's m_prefill_word to advance all the way to
        // m_count while the remaining `live_cross` blocks stay untouched.
        std::thread([&] {
            constexpr std::size_t kMore = 5000;
            std::vector<void *> c_blocks(kMore, nullptr);
            for(std::size_t j = 0; j < kMore; ++j) {
                void *p = kame_pool_malloc(kSz);
                CHECK(p != nullptr, "adopter malloc failed");
                if(p) stamp_block(p, kSz, 99u, j);
                c_blocks[j] = p;
            }
            // Verify A's still-live blocks were never handed out twice or overwritten.
            std::size_t bad_a = 0, bad_c = 0;
            for(std::size_t i = 2; i < kPartial; i += 4) {
                if(live_cross[i] &&
                   !verify_block(live_cross[i], kSz, static_cast<std::size_t>(round), i))
                    ++bad_a;
            }
            for(std::size_t j = 0; j < kMore; ++j) {
                if(c_blocks[j] && !verify_block(c_blocks[j], kSz, 99u, j))
                    ++bad_c;
                kame_pool_free(c_blocks[j]);
            }
            CHECK(bad_a == 0, "round %d: %zu live producer blocks overwritten", round, bad_a);
            CHECK(bad_c == 0, "round %d: %zu adopter blocks overwritten", round, bad_c);
        }).join();

        for(void *p : live_cross)
            if(p) kame_pool_free(p);
    }
    kame_pool_trim();
}

// 3. 32 concurrent sparse threads × 17 FS=true buckets RSS bound.
void scenario_sparse_threads_rss() {
    std::printf("[3/4] 32 concurrent sparse threads x 17 FS=true buckets RSS bound\n");
    kame_pool_trim();
    const std::size_t rss_before = current_rss_bytes();

    constexpr unsigned kThreads = 32;
    std::mutex mtx;
    std::condition_variable cv_ready, cv_go;
    unsigned ready = 0;
    bool release = false;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for(unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            void *ptrs[17] = {};
            for(std::size_t b = 0; b < 17; ++b) {
                ptrs[b] = kame_pool_malloc(kFsTrueSizes[b]);
                if(ptrs[b])
                    stamp_block(ptrs[b], kFsTrueSizes[b], b, t);
            }
            {
                std::unique_lock<std::mutex> lk(mtx);
                if(++ready == kThreads)
                    cv_ready.notify_one();
                cv_go.wait(lk, [&] { return release; });
            }
            for(std::size_t b = 0; b < 17; ++b) {
                if(ptrs[b]) {
                    CHECK(verify_block(ptrs[b], kFsTrueSizes[b], b, t),
                          "sparse thread %u bucket %zu corrupted", t, b);
                    kame_pool_free(ptrs[b]);
                }
            }
        });
    }

    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_ready.wait(lk, [&] { return ready == kThreads; });
    }
    const std::size_t rss_peak = current_rss_bytes();
    {
        std::lock_guard<std::mutex> lk(mtx);
        release = true;
    }
    cv_go.notify_all();
    for(auto &th : workers)
        th.join();
    kame_pool_trim();

    if(rss_before > 0 && rss_peak >= rss_before) {
        const std::size_t delta_mib = (rss_peak - rss_before) >> 20;
        std::printf("      sparse 32T x 17 buckets RSS delta = %zu MiB "
                    "(before=%zu MiB, live=%zu MiB)\n",
                    delta_mib, rss_before >> 20, rss_peak >> 20);
        // Full-chunk pre-fill dirties ~192 MiB across 32 * 17 chunks; 1-page
        // watermark dirties ~2.2 MiB (4K pages) or ~9.6 MiB (16K pages) plus
        // thread stacks/TLS.  64 MiB leaves > 3x margin while catching any
        // regression back to full-chunk pre-fill.
        CHECK(delta_mib < 64u,
              "sparse 32T RSS delta too high: %zu MiB (expected < 64 MiB)",
              delta_mib);
    }
}

// 4. kame_pool_prewarm holds > 128 blocks simultaneously and leaves RT clean.
void scenario_prewarm_full_capacity() {
    std::printf("[4/4] kame_pool_prewarm > 128 blocks holds capacity for RT section\n");
    std::thread([] {
        const std::size_t sizes[2]  = {64, 1024};
        const unsigned    counts[2] = {600, 600};
        int rc = kame_pool_prewarm(sizes, counts, 2);
        CHECK(rc == 0, "kame_pool_prewarm returned %d", rc);

        kame_pool_rt_reset_counters();
        kame_pool_set_realtime_thread(1);
        std::vector<void *> v64(500, nullptr), v1k(500, nullptr);
        for(std::size_t i = 0; i < 500; ++i) {
            v64[i] = kame_pool_malloc(64);
            v1k[i] = kame_pool_malloc(1024);
            if(v64[i]) std::memset(v64[i], 0x11, 64);
            if(v1k[i]) std::memset(v1k[i], 0x22, 1024);
        }
        for(std::size_t i = 0; i < 500; ++i) {
            kame_pool_free(v64[i]);
            kame_pool_free(v1k[i]);
        }
        const unsigned long long viol = kame_pool_rt_violations();
        kame_pool_set_realtime_thread(0);
        kame_pool_rt_drain();
        CHECK(viol == 0ull,
              "RT section after kame_pool_prewarm(600) had %llu RT violations",
              viol);
    }).join();
}

} // namespace

int main() {
    scenario_expand_all_buckets();
    scenario_partial_orphan_adopt();
    scenario_sparse_threads_rss();
    scenario_prewarm_full_capacity();
    if(g_failures != 0) {
        std::fprintf(stderr, "alloc_watermark_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("alloc_watermark_test: PASSED\n");
    return 0;
}
