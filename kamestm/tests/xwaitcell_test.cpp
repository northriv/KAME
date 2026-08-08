// Behavioural test for XWaitCell, the timed wait-on-address primitive that
// Node::NegotiationCounter::negotiate_sleep parks on.
//
// The backend is chosen at COMPILE time — __ulock on macOS, futex on Linux,
// std mutex + condvar everywhere else — and nothing else in the suite exercises
// the primitive directly, so a backend can differ from the others in a way no
// transaction test would localise.  This pins the four properties
// negotiate_sleep actually relies on, and is written to pass identically on all
// three: build it with -DKAME_XWAITCELL_ULOCK=0 / -DKAME_XWAITCELL_FUTEX=0 to
// check the fallback on a platform that would otherwise skip it.
//
// Deliberately not a timing benchmark: the only timing assertions are
// one-sided and generously slack, because CI hosts are shared.
#include "support_standalone.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "xwaitcell.h"

static int g_failed = 0;

#define CHECK(cond, ...) do { \
        if( !(cond)) { \
            std::printf("FAILED %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__); std::printf("\n"); \
            g_failed++; \
        } \
    } while(0)

using clk = std::chrono::steady_clock;
static long long elapsed_us(clk::time_point t0) {
    return (long long)std::chrono::duration_cast<std::chrono::microseconds>(
        clk::now() - t0).count();
}

int main() {
    std::printf("== xwaitcell_test == backend: %s (ULOCK=%d FUTEX=%d)"
                "  sizeof(XWaitCell)=%u\n",
                KAME_XWAITCELL_ULOCK ? "__ulock" :
                    (KAME_XWAITCELL_FUTEX ? "futex" : "mutex+condvar"),
                (int)KAME_XWAITCELL_ULOCK, (int)KAME_XWAITCELL_FUTEX,
                (unsigned)sizeof(XWaitCell));

    // (1) A wait on the CURRENT generation must block for roughly the whole
    //     timeout and report "not woken".  This is the ordinary sleep in
    //     negotiate_sleep when nobody is contending for the same linkage.
    {
        XWaitCell c;
        auto t0 = clk::now();
        bool woken = c.wait(c.gen(), 50000);
        long long dt = elapsed_us(t0);
        CHECK( !woken, "timeout must report not-woken");
        CHECK(dt >= 40000, "timeout returned after only %lld us; it must not "
                           "wake early (the caller treats that as a signal)", dt);
    }

    // (2) usec == 0 means "poll", NOT "wait forever" — negotiate_sleep hands a
    //     clamped budget that can round to zero, and a block there would burn
    //     the whole wait budget in one call.
    {
        XWaitCell c;
        auto t0 = clk::now();
        CHECK( !c.wait(c.gen(), 0), "zero-length wait on the current "
                                    "generation must report not-woken");
        CHECK(elapsed_us(t0) < 20000, "zero-length wait blocked");
        uint32_t g = c.gen();
        c.wake_one();
        CHECK(c.wait(g, 0), "zero-length wait on a stale generation must "
                            "report woken");
    }

    // (3) The lost-wakeup window: a wake that lands after gen() but before
    //     wait() must NOT put the caller to sleep.  This is the whole reason
    //     the mutex-less backends are sound, so it is the property most worth
    //     pinning — a backend that got it wrong would hang here, not misreport.
    {
        XWaitCell c;
        uint32_t g = c.gen();
        c.wake_one();                       // the racing waker
        auto t0 = clk::now();
        bool woken = c.wait(g, 5000000);    // 5 s: a regression hangs visibly
        long long dt = elapsed_us(t0);
        CHECK(woken, "a stale generation must report woken");
        CHECK(dt < 100000, "a stale generation slept for %lld us — the "
                           "value-compare did not close the lost-wakeup "
                           "window", dt);
    }

    // (4) A real cross-thread wake arrives, and arrives promptly.
    {
        XWaitCell c;
        uint32_t g = c.gen();
        std::atomic<bool> waiting{false};
        std::thread waker([&]{
            while( !waiting.load(std::memory_order_acquire))
                std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            c.wake_one();
        });
        waiting.store(true, std::memory_order_release);
        auto t0 = clk::now();
        bool woken = c.wait(g, 5000000);
        long long dt = elapsed_us(t0);
        waker.join();
        CHECK(woken, "a cross-thread wake_one() must report woken");
        CHECK(dt < 2000000, "a cross-thread wake took %lld us", dt);
        CHECK(c.gen() != g, "wake_one() must advance the generation");
    }

    // (5) Many wakes against many waiters: each wake_one() releases at least
    //     one sleeper, and no sleeper is stranded.  negotiate_sleep's
    //     notify_n_contenders does exactly this against a slot array.
    {
        constexpr int N = 8;
        XWaitCell cells[N];
        std::atomic<int> awake{0};
        std::thread th[N];
        for(int i = 0; i < N; i++) {
            th[i] = std::thread([&, i]{
                if(cells[i].wait(cells[i].gen(), 5000000))
                    awake.fetch_add(1, std::memory_order_relaxed);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for(int i = 0; i < N; i++) cells[i].wake_one();
        for(int i = 0; i < N; i++) th[i].join();
        CHECK(awake.load() == N, "only %d of %d sleepers were woken",
              awake.load(), N);
    }

    std::printf(g_failed ? "== FAILED (%d) ==\n" : "== PASSED ==\n", g_failed);
    return g_failed ? 1 : 0;
}
