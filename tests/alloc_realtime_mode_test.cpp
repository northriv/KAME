// (§30) Smoke-test for kame_pool_set_realtime_mode().
//
// Verifies the realtime-mode preset toggles the three background
// maintenance knobs and restores them on `enable == 0`:
//
//   (1) Lazy-drain interval (§28.1) goes effectively infinite when
//       realtime is ON, snaps back to the 10 ms default when OFF.
//   (2) Auto-tune (§28.3) is locked out while realtime is ON — even
//       after a deliberate LRC_MMAP push (which would otherwise trigger
//       the one-shot munmap probe), the interval stays at its huge
//       realtime-mode value, NOT overwritten.
//   (3) Thread-exit reclaim (§21) — no direct C API to query, but the
//       preset is documented to flip the same flag
//       `kame_pool_set_thread_exit_reclaim` drives.  We verify the set
//       call doesn't error and threads spawn/exit cleanly under
//       realtime-mode (the failure mode would be a crash or stall in
//       `release_dll_chunks_for_thread`).
//
// What this test does NOT verify:
//   * Actual elimination of munmap/madvise syscalls in the steady state
//     — would need strace/dtrace.  The presence of the flag at the
//     value-level is the contract; the implementation's use of it is
//     covered by the lazy-drain and chunk-release ctests.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <cstdio>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

int main() {
    // (1) Default lazy-drain interval is 10 ms.
    {
        unsigned int def_ms = kame_pool_get_lazy_drain_interval_ms();
        CHECK(def_ms == 10u,
              "default lazy-drain interval not 10 ms (got %u)", def_ms);
        std::printf("  [ok] default lazy interval = %u ms\n", def_ms);
    }

    // (2) Realtime ON → lazy interval becomes huge (> 11 days).
    {
        kame_pool_set_realtime_mode(1);
        unsigned int rt_ms = kame_pool_get_lazy_drain_interval_ms();
        CHECK(rt_ms > 1000u * 1000u * 1000u,
              "realtime-ON lazy interval not huge (got %u ms; want > 1e9)",
              rt_ms);
        std::printf("  [ok] realtime ON lazy interval = %u ms\n", rt_ms);

        // (3) Do some LRC_MMAP-tier work; auto-tune must NOT fire (it
        // would otherwise overwrite the interval with `20 × munmap_ns`).
        std::vector<void *> ptrs;
        for(int i = 0; i < 64; ++i) {
            void *p = kame_pool_malloc(8u << 20);   // 8 MiB → MMAP tier
            if(p) ptrs.push_back(p);
        }
        for(void *p : ptrs) kame_pool_free(p);
        unsigned int after_ms = kame_pool_get_lazy_drain_interval_ms();
        CHECK(after_ms == rt_ms,
              "auto-tune leaked through realtime-ON "
              "(was %u ms; now %u ms)", rt_ms, after_ms);
        std::printf("  [ok] auto-tune locked under realtime mode "
                    "(interval still %u ms)\n", after_ms);

        // (4) Threads spawn / alloc / free / exit cleanly with the
        // thread-exit madvise off.  An incorrect realtime_mode could
        // leave a stale flag that crashes `release_dll_chunks_for_thread`.
        auto worker = []{
            for(int i = 0; i < 64; ++i) {
                void *p = kame_pool_malloc(1024);
                if(p) kame_pool_free(p);
            }
        };
        std::vector<std::thread> ts;
        for(int t = 0; t < 4; ++t) ts.emplace_back(worker);
        for(auto &t : ts) t.join();
        std::printf("  [ok] thread teardown under realtime mode\n");
    }

    // (5) Realtime OFF → defaults restored.
    {
        kame_pool_set_realtime_mode(0);
        unsigned int reset_ms = kame_pool_get_lazy_drain_interval_ms();
        CHECK(reset_ms == 10u,
              "realtime OFF didn't restore default 10 ms (got %u)",
              reset_ms);
        std::printf("  [ok] realtime OFF restored 10 ms default\n");
    }

    // (6) Re-toggling is idempotent — ON twice in a row stays huge,
    // OFF twice in a row stays at default.
    {
        kame_pool_set_realtime_mode(1);
        kame_pool_set_realtime_mode(1);
        unsigned int rt2 = kame_pool_get_lazy_drain_interval_ms();
        CHECK(rt2 > 1000u * 1000u * 1000u,
              "double ON didn't preserve huge (got %u)", rt2);
        kame_pool_set_realtime_mode(0);
        kame_pool_set_realtime_mode(0);
        unsigned int off2 = kame_pool_get_lazy_drain_interval_ms();
        CHECK(off2 == 10u, "double OFF didn't restore default (got %u)", off2);
        std::printf("  [ok] toggling is idempotent\n");
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
