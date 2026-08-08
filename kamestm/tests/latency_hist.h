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
// =====================================================================
// latency_hist.h — the allocation-free latency histogram shared by the
// kamestm test harnesses.
//
// Lifted verbatim out of transaction_latency_bench.cpp when
// transaction_priority_mixed_test needed the same thing: a distribution
// under the deployment's ROLE mix, not just a throughput count.  One copy,
// because two would drift and the percentile arithmetic is exactly the
// part a reader has to trust.
//
// 1 ns resolution below 64 ns, then 4 buckets per octave.  O(1) memory and
// no allocation, so the harness cannot perturb the allocator underneath
// the STM it is measuring.
// =====================================================================
#ifndef KAMESTM_TESTS_LATENCY_HIST_H
#define KAMESTM_TESTS_LATENCY_HIST_H

#include <chrono>
#include <cstdint>
#include <cstring>

//! Monotonic nanoseconds.
static inline std::uint64_t now_ns() {
    return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum { HB = 256 };

struct Hist {
    std::uint64_t bucket[HB];
    std::uint64_t n, max, sum;
    void reset() { std::memset(this, 0, sizeof(*this)); }
    static unsigned idx(std::uint64_t v) {
        if(v < 64) return (unsigned)v;
        unsigned oct = 63u - (unsigned)__builtin_clzll(v);
        unsigned i = 64u + (oct - 6u) * 4u + (unsigned)((v >> (oct - 2)) & 3u);
        return i < (unsigned)HB ? i : (unsigned)HB - 1u;
    }
    static std::uint64_t value(unsigned i) {
        if(i < 64) return i;
        unsigned oct = 6u + (i - 64u) / 4u, frac = (i - 64u) % 4u;
        return (std::uint64_t)(4u + frac + 1u) << (oct - 2);
    }
    void add(std::uint64_t v) {
        bucket[idx(v)]++; n++; sum += v; if(v > max) max = v;
    }
    void merge(const Hist &o) {
        for(unsigned i = 0; i < HB; i++) bucket[i] += o.bucket[i];
        n += o.n; sum += o.sum; if(o.max > max) max = o.max;
    }
    std::uint64_t pct(double p) const {
        if( !n) return 0;
        std::uint64_t want = (std::uint64_t)(p * (double)n), acc = 0;
        if(want >= n) want = n - 1;
        for(unsigned i = 0; i < HB; i++)
            if((acc += bucket[i]) > want) return value(i);
        return max;
    }
    //! A percentile is only meaningful with >= 10 samples beyond it.
    bool supports(double p) const { return (double)n * (1.0 - p) >= 10.0; }
    //! Commits at or above `v` ns — used for the "reached the sleep" estimate.
    std::uint64_t at_or_above(std::uint64_t v) const {
        std::uint64_t acc = 0;
        for(unsigned i = 0; i < HB; i++) if(value(i) >= v) acc += bucket[i];
        return acc;
    }
};

#endif // KAMESTM_TESTS_LATENCY_HIST_H
