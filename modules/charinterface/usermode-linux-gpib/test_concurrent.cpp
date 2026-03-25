/*
 * test_concurrent.cpp — tests for continuous query and concurrent operations.
 *
 * Mock mode (default, no hardware required):
 *   A mock gpib_interface is injected via NiGpibDriver::openForTest() so the
 *   full send/read/query/serialPoll paths run against in-process stubs.
 *
 *   Build (macOS, libusb via MacPorts):
 *     clang++ -I. -Ilinux-gpib -I/opt/local/include -std=gnu++17 \
 *         -Wno-unused-function -Wno-visibility \
 *         test_concurrent.cpp NiGpibDriver.cpp gpib_stubs.c \
 *         -L/opt/local/lib -lusb-1.0 -lpthread -o test_concurrent
 *     ./test_concurrent
 *
 * Real hardware mode (compile with -DREAL_HARDWARE):
 *   Uses NiGpibDriver::open() against an actual NI USB-GPIB adapter.
 *   Instrument address is controlled by HW_ADDR (default 12).
 *
 *   Build:
 *     clang  -I. -Ilinux-gpib -I/opt/local/include -Wno-unused-function \
 *         -Wno-visibility -std=gnu11 \
 *         -c linux-gpib/ni_usb_gpib.c gpib_stubs.c
 *     clang++ -I. -Ilinux-gpib -I/opt/local/include -Wno-unused-function \
 *         -Wno-visibility -std=gnu++17 -DREAL_HARDWARE \
 *         -c test_concurrent.cpp NiGpibDriver.cpp
 *     clang++ test_concurrent.o NiGpibDriver.o ni_usb_gpib.o gpib_stubs.o \
 *         -L/opt/local/lib -lusb-1.0 -lpthread -o test_concurrent_hw
 *     ./test_concurrent_hw                   # addr 12
 *     HW_ADDR=5 ./test_concurrent_hw         # override address
 */

/* Include C++ standard library headers before NiGpibDriver.h.
 * osx_compat.h (pulled in by NiGpibDriver.h inside an extern "C" block) no
 * longer includes <stdatomic.h> in C++ mode, so there is no macro/function
 * clash with <atomic>.  But including <atomic> and <thread> first is still
 * good practice to ensure the C++ headers see a clean namespace. */
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "NiGpibDriver.h"

/* ── stubs replacing ni_usb_gpib.c ─────────────────────────────────────────
 * ni_module_init / ni_module_exit are normally provided by ni_usb_gpib.c.
 * In mock mode we define trivial no-ops so the test links without that TU.
 * In real hardware mode (-DREAL_HARDWARE) the actual symbols from
 * ni_usb_gpib.c are linked in, so these stubs must not be defined.
 */
#ifndef REAL_HARDWARE
extern "C" {
    int  ni_module_init(void) { return 0; }
    void ni_module_exit(void) {}
}
#endif /* !REAL_HARDWARE */

/* ── mock gpib_interface ─────────────────────────────────────────────────── */

static std::atomic<int> g_cmd_calls{0};
static std::atomic<int> g_write_calls{0};
static std::atomic<int> g_read_calls{0};

/* The fixed response every mock_read() returns (before stripCrLf). */
static const char MOCK_RESP[] = "MOCK,1.0\n";
/* Expected result after NiGpibDriver strips the trailing '\n'. */
static const char EXPECT_QUERY[] = "MOCK,1.0";
/* serialPoll reads 1 byte → first byte of MOCK_RESP. */
static const uint8_t EXPECT_STB = static_cast<uint8_t>(MOCK_RESP[0]); /* 'M' */

static int mock_attach(struct gpib_board *, const struct gpib_board_config *)
{
    return 0;
}
static void mock_detach(struct gpib_board *) {}

static int mock_command(struct gpib_board *, u8 *, size_t len, size_t *bw)
{
    ++g_cmd_calls;
    *bw = len;
    return 0;
}

static int mock_write(struct gpib_board *, u8 *, size_t len, int, size_t *bw)
{
    ++g_write_calls;
    *bw = len;
    return 0;
}

static int mock_read(struct gpib_board *, u8 *buf, size_t len,
                     int *end, size_t *br)
{
    ++g_read_calls;
    size_t n = (len < sizeof(MOCK_RESP) - 1) ? len : sizeof(MOCK_RESP) - 1;
    memcpy(buf, MOCK_RESP, n);
    *end = 1;
    *br  = n;
    return 0;
}

static int  mock_enable_eos(struct gpib_board *, u8, int) { return 0; }
static void mock_disable_eos(struct gpib_board *)          {}
static void mock_remote_enable(struct gpib_board *, int)   {}
static void mock_interface_clear(struct gpib_board *, int) {}

static unsigned int mock_update_status(struct gpib_board *,
                                       unsigned int)        { return 0; }

/* The single shared mock interface.  All function pointers not explicitly set
 * remain NULL; they are not called by the code paths exercised here. */
static struct gpib_interface g_mock_iface;

static void init_mock_iface(void)
{
    memset(&g_mock_iface, 0, sizeof(g_mock_iface));
    g_mock_iface.name            = (char *)"mock";
    g_mock_iface.attach          = mock_attach;
    g_mock_iface.detach          = mock_detach;
    g_mock_iface.command         = mock_command;
    g_mock_iface.write           = mock_write;
    g_mock_iface.read            = mock_read;
    g_mock_iface.enable_eos      = mock_enable_eos;
    g_mock_iface.disable_eos     = mock_disable_eos;
    g_mock_iface.remote_enable   = mock_remote_enable;
    g_mock_iface.interface_clear = mock_interface_clear;
    g_mock_iface.update_status   = mock_update_status;
}

/* ── test A: continuous query ────────────────────────────────────────────── */
/*
 * A single NiGpibDriver runs N sequential query() calls and verifies that
 * each response matches the mock's canned reply.  Exercises the full
 * send→command→write→enable_eos→command→read→disable_eos→stripCrLf pipeline
 * without any real hardware or USB involvement.
 */
static bool test_continuous_query(int n)
{
    printf("  test_continuous_query: %d iterations\n", n);

    NiGpibDriver drv;
    drv.openForTest(&g_mock_iface);

    int failures = 0;
    for (int i = 0; i < n; i++) {
        std::string resp;
        try {
            resp = drv.query(1, "*IDN?", "\n");
        } catch (const std::exception &ex) {
            fprintf(stderr, "    [%d] exception: %s\n", i, ex.what());
            ++failures;
            continue;
        }
        if (resp != EXPECT_QUERY) {
            fprintf(stderr, "    [%d] expected \"%s\", got \"%s\"\n",
                    i, EXPECT_QUERY, resp.c_str());
            ++failures;
        }
    }

    printf("    %d/%d correct\n", n - failures, n);
    return failures == 0;
}

/* ── test B: concurrent query ────────────────────────────────────────────── */
/*
 * T threads each create their own NiGpibDriver (one per thread, matching the
 * single-master GPIB model) and issue Q queries concurrently.  Because each
 * driver owns its own gpib_board with its own mutexes/waitqueues, and the
 * mock functions are stateless (apart from atomic counters), this exercises
 * concurrent driver lifecycle (open/query/close) without hardware.
 */
static bool test_concurrent_query(int threads, int queries_per_thread)
{
    printf("  test_concurrent_query: %d threads × %d queries\n",
           threads, queries_per_thread);

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t]() {
            NiGpibDriver drv;
            drv.openForTest(&g_mock_iface);
            for (int i = 0; i < queries_per_thread; i++) {
                try {
                    std::string resp = drv.query(1, "*IDN?", "\n");
                    if (resp != EXPECT_QUERY) {
                        fprintf(stderr,
                                "    thread %d iter %d: got \"%s\"\n",
                                t, i, resp.c_str());
                        ++failures;
                    }
                } catch (const std::exception &ex) {
                    fprintf(stderr,
                            "    thread %d iter %d: exception: %s\n",
                            t, i, ex.what());
                    ++failures;
                }
            }
        });
    }
    for (auto &th : workers) th.join();

    int total = threads * queries_per_thread;
    printf("    %d/%d correct\n", total - failures.load(), total);
    return failures.load() == 0;
}

/* ── test C: concurrent serial poll ─────────────────────────────────────── */
/*
 * T threads each open a driver and call serialPoll() concurrently.
 * serialPoll issues SPE/MTA/MLA command bytes, reads one byte (the STB), then
 * SPD/UNL/UNT.  The mock read returns the first byte of MOCK_RESP ('M').
 */
static bool test_concurrent_serial_poll(int threads)
{
    printf("  test_concurrent_serial_poll: %d threads\n", threads);

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t]() {
            NiGpibDriver drv;
            drv.openForTest(&g_mock_iface);
            try {
                uint8_t stb = drv.serialPoll(1);
                if (stb != EXPECT_STB) {
                    fprintf(stderr,
                            "    thread %d: expected STB=0x%02X, got 0x%02X\n",
                            t, EXPECT_STB, stb);
                    ++failures;
                }
            } catch (const std::exception &ex) {
                fprintf(stderr,
                        "    thread %d: serialPoll exception: %s\n",
                        t, ex.what());
                ++failures;
            }
        });
    }
    for (auto &th : workers) th.join();

    printf("    %d/%d correct\n",
           threads - failures.load(), threads);
    return failures.load() == 0;
}

/* ── real hardware tests ────────────────────────────────────────────────── */
#ifdef REAL_HARDWARE

using us_t = long long;
using Clock = std::chrono::steady_clock;

static us_t elapsed_us(Clock::time_point t0)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - t0).count();
}

/*
 * Test D — continuous query on real hardware.
 *
 * Sends `cmd` to `addr` N times via query() and prints per-query timing
 * statistics.  Verifies that each response is non-empty and optionally checks
 * that it contains `expect` (if non-null).
 */
static bool test_hw_continuous_query(NiGpibDriver &drv, int addr,
                                     const char *cmd, int n,
                                     const char *expect = nullptr)
{
    printf("  test_hw_continuous_query: %d × \"%s\" to addr %d\n",
           n, cmd, addr);

    us_t total = 0, mn = LLONG_MAX, mx = 0;
    int failures = 0;

    for (int i = 0; i < n; i++) {
        auto t0 = Clock::now();
        std::string resp;
        try {
            resp = drv.query(addr, cmd);
        } catch (const std::exception &ex) {
            fprintf(stderr, "    [%d] exception: %s\n", i, ex.what());
            ++failures;
            continue;
        }
        us_t dt = elapsed_us(t0);
        total += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;

        if (resp.empty()) {
            fprintf(stderr, "    [%d] empty response (%.1f ms)\n",
                    i, dt / 1000.0);
            ++failures;
        } else if (expect && resp.find(expect) == std::string::npos) {
            fprintf(stderr, "    [%d] \"%s\" missing \"%s\" (%.1f ms)\n",
                    i, resp.c_str(), expect, dt / 1000.0);
            ++failures;
        } else {
            printf("    [%2d] %s  (%.1f ms)\n", i, resp.c_str(), dt / 1000.0);
        }
    }

    int ok = n - failures;
    printf("    %d/%d ok  |  min %.1f ms  avg %.1f ms  max %.1f ms  total %.2f s\n",
           ok, n,
           mn / 1000.0,
           (n > 0 ? total / (double)n : 0.0) / 1000.0,
           mx / 1000.0,
           total / 1e6);
    return failures == 0;
}

/*
 * Test E — serial poll on real hardware, N times.
 *
 * Measures per-call timing.  The STB value is printed but not validated
 * (instrument state is unknown).
 */
static bool test_hw_serial_poll(NiGpibDriver &drv, int addr, int n)
{
    printf("  test_hw_serial_poll: %d × serialPoll(%d)\n", n, addr);

    us_t total = 0, mn = LLONG_MAX, mx = 0;
    int failures = 0;
    uint8_t last_stb = 0;

    for (int i = 0; i < n; i++) {
        auto t0 = Clock::now();
        uint8_t stb = 0;
        try {
            stb = drv.serialPoll(addr);
        } catch (const std::exception &ex) {
            fprintf(stderr, "    [%d] exception: %s\n", i, ex.what());
            ++failures;
            continue;
        }
        us_t dt = elapsed_us(t0);
        total += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
        last_stb = stb;
    }

    int ok = n - failures;
    printf("    %d/%d ok  last STB=0x%02X  |  "
           "min %.1f ms  avg %.1f ms  max %.1f ms\n",
           ok, n, last_stb,
           mn / 1000.0,
           (n > 0 ? total / (double)n : 0.0) / 1000.0,
           mx / 1000.0);
    return failures == 0;
}

#endif /* REAL_HARDWARE */

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
#ifdef REAL_HARDWARE
    const char *addr_env = getenv("HW_ADDR");
    int hw_addr = addr_env ? atoi(addr_env) : 12;

    printf("=== GPIB real hardware tests (addr %d) ===\n\n", hw_addr);

    NiGpibDriver hw_drv;
    if (!hw_drv.open()) {
        fprintf(stderr, "Failed to open NI USB-GPIB adapter.\n");
        return 1;
    }

    bool all = true;
    bool r;

    printf("[D] Continuous *IDN? query (20 iterations)\n");
    r = test_hw_continuous_query(hw_drv, hw_addr, "*IDN?", 20, nullptr);
    printf("    -> %s\n\n", r ? "PASS" : "FAIL");
    all &= r;

    printf("[E] Serial poll (10 iterations)\n");
    r = test_hw_serial_poll(hw_drv, hw_addr, 10);
    printf("    -> %s\n\n", r ? "PASS" : "FAIL");
    all &= r;

    printf("\n=== %s ===\n", all ? "ALL PASS" : "SOME TESTS FAILED");
    return all ? 0 : 1;

#else /* mock mode */

    init_mock_iface();

    printf("=== GPIB mock tests ===\n\n");

    bool all = true;
    bool r;

    printf("[A] Continuous query\n");
    r = test_continuous_query(200);
    printf("    -> %s\n\n", r ? "PASS" : "FAIL");
    all &= r;

    printf("[B] Concurrent query (8 threads × 25 queries)\n");
    r = test_concurrent_query(8, 25);
    printf("    -> %s\n\n", r ? "PASS" : "FAIL");
    all &= r;

    printf("[C] Concurrent serial poll (8 threads)\n");
    r = test_concurrent_serial_poll(8);
    printf("    -> %s\n\n", r ? "PASS" : "FAIL");
    all &= r;

    printf("Total mock read() calls: %d\n", g_read_calls.load());
    printf("Total mock write() calls: %d\n", g_write_calls.load());
    printf("Total mock command() calls: %d\n", g_cmd_calls.load());
    printf("\n=== %s ===\n", all ? "ALL PASS" : "SOME TESTS FAILED");
    return all ? 0 : 1;

#endif /* REAL_HARDWARE */
}
