/*
 * latency_floor — what the tail costs with NO STM in it.
 *
 * The control every latency number in this directory needs and did not have.
 * `transaction_priority_mixed_test` and `transaction_latency_bench` time a
 * commit between two `now_ns()` calls; whatever interrupts the machine between
 * those two reads lands in the histogram as if the STM had spent it.  Nothing
 * said how large that share was, so a residue got attributed to `commit()` and
 * an instrument was nearly built to hunt it — see the 2026-08 note at the end.
 *
 * This runs the identical clock and the identical histogram with the STM
 * removed, so the two can be subtracted:
 *
 *   clk   two back-to-back now_ns() with NOTHING between.  Everything here is
 *         the clock read itself plus the machine interrupting between them.
 *   work  the same, with the acq lambda's arithmetic between.  Equal to clk
 *         => the machine; larger => the work.
 *
 * The two arms are interleaved in one loop so they see the same conditions.
 *
 * REPORTED AS EVENTS PER SECOND, not only percentiles, because percentiles are
 * NOT comparable across arms with different iteration rates: this loop runs
 * ~10^7 iterations/s and a record commit ~10^5, so one machine event per second
 * is p99.9999 here and p99.99 there.  A tail is attributable only after both
 * are expressed per unit time.
 *
 * Also prints the clocksource, because a fallback to hpet/acpi_pm makes every
 * `now_ns()` a syscall of microseconds and invalidates every latency figure
 * this directory produces.  (It shows up in the MEDIAN, not the tail — a p50
 * far below a microsecond is already proof that the TSC vDSO path is in use.)
 *
 * Not a ctest: like transaction_latency_bench, absolute latencies are a
 * property of the host, so this is run deliberately and quoted with the host.
 *
 * Usage:  latency_floor [seconds]        (default 20)
 *
 * On a realtime host, run it the way the thing it is a control for runs:
 *
 *     sudo chrt -f 20 taskset -c <isolated cpu> ./latency_floor 600
 *
 * 2026-08, the measurement that prompted this file: on a shared 4-CPU
 * container the `clk` arm — two clock reads, no STM whatsoever — produced
 * 1.05 events/s over 50 us and 0.05 events/s over 95 us, i.e. ~6 events over
 * 95 us per 120 s.  The PREEMPT_RT host's 6,568,736-commit run reported
 * exactly ONE sample over 95 us in 120 s.  A bare pair of clock reads was
 * therefore noisier than the commit whose MAX was being investigated, and the
 * "95 us ceiling" was a machine event that happened to land inside a timed
 * region.  Run this before concluding anything about a tail.
 */
/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
#include "latency_hist.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

//! Thresholds reported as a rate.  The last two are the ones a commit
//! histogram's MAX usually sits on.
static const std::uint64_t kThresh[] = {1000, 10000, 50000, 95000};
static const char *kThreshName[] = {"1us", "10us", "50us", "95us"};

static void report(const char *tag, const Hist &h, double secs) {
    std::printf("  %-5s n=%-12llu mean=%-5llu p50=%-5llu p99=%-6llu "
                "p99.9=%-7llu p99.99=%-9llu MAX=%llu ns\n", tag,
        (unsigned long long)h.n,
        (unsigned long long)(h.n ? h.sum / h.n : 0),
        (unsigned long long)h.pct(0.50),  (unsigned long long)h.pct(0.99),
        (unsigned long long)h.pct(0.999), (unsigned long long)h.pct(0.9999),
        (unsigned long long)h.max);
    std::printf("        events/s over: ");
    for(unsigned i = 0; i < sizeof(kThresh) / sizeof(kThresh[0]); ++i)
        std::printf(" %s=%.3f", kThreshName[i],
                    (double)h.at_or_above(kThresh[i]) / secs);
    std::printf("\n");
}

int main(int argc, char **argv) {
    long secs = (argc > 1) ? std::atol(argv[1]) : 20;
    if(secs < 1) secs = 1;

    char cs[64] = "?";
    if(FILE *f = std::fopen("/sys/devices/system/clocksource/clocksource0/"
                            "current_clocksource", "r")) {
        if(std::fgets(cs, sizeof(cs), f)) cs[std::strcspn(cs, "\n")] = 0;
        std::fclose(f);
    }
    std::printf("latency_floor: %ld s, clocksource=%s\n", secs, cs);

    Hist hclk, hwork;
    hclk.reset(); hwork.reset();
    //! volatile so the arithmetic cannot be optimised away — without it the
    //! `work` arm silently becomes a second copy of `clk`.
    volatile long sink = 0;

    const std::uint64_t t_start = now_ns();
    const std::uint64_t t_end = t_start + (std::uint64_t)secs * 1000000000ull;
    //! Inner batch so the loop condition's own clock read is amortised rather
    //! than doubling the sample count of the arm it precedes.
    while(now_ns() < t_end) {
        for(int r = 0; r < 1000; ++r) {
            std::uint64_t a = now_ns();
            hclk.add(now_ns() - a);
            std::uint64_t b = now_ns();
            for(int i = 0; i < 5; ++i) sink += i;
            hwork.add(now_ns() - b);
        }
    }
    const double el = (double)(now_ns() - t_start) / 1e9;

    report("clk",  hclk,  el);
    report("work", hwork, el);
    std::printf("  (clk ~= work => the tail is the machine, not the work.  "
                "Compare the events/s lines, NOT the percentiles, against a "
                "commit histogram: the iteration rates differ by ~100x.)\n");
    return 0;
}
