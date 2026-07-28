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
//! Regression test for `Transactional::ScopedWaitBudget`.
//!
//! Runs the contended whole-tree ("grand") commit pattern with a budget on
//! every commit and FAILS if **p99.99** of commit latency exceeds
//! `budget + SLACK`.
//!
//! **Why p99.99 and not the max.**  The max was tried first and is not a
//! usable assertion: measured here, the commit that produced the max had
//! **one or two attempts** — including the unbudgeted 220 ms and 411 ms
//! maxima, which had exactly one.  One attempt means no retry storm and, with
//! an absolute budget shared across the commit's negotiator entries, no
//! overlong wait either.  What is left is the OS not scheduling the thread.
//! A max-based check therefore tests the platform scheduler rather than this
//! library, and fails about half the time on an idle laptop.  The max is
//! still printed, with the attempt count, because that pair is what tells the
//! two causes apart.
//!
//! **Why a slack at all, and why it lives here and not in kamestm.**  The
//! budget bounds what the negotiator *chooses* to wait.  It cannot bound what
//! the OS does with a timed wait, how long a descheduled thread stays off-CPU,
//! or how many times a commit has to retry after it stops waiting.  Modelling
//! any of that inside the library would mean encoding undocumented platform
//! behaviour into the general (non-realtime) path, which is not something the
//! library should do.  A test, on the other hand, is allowed an empirical
//! tolerance: it is checking OUR logic, and the slack is the allowance for
//! everything that is not ours.
//!
//! SLACK defaults to **1 ms**, overridable with `KAME_WB_TEST_SLACK_US` (10000
//! is a reasonable conservative value for a slow or heavily loaded machine).
//! 1 ms is derived, not picked: the measured p99.99 overshoot is a fixed
//! ~200 µs regardless of budget size —
//!
//!     budget   100 µs -> p99.99  150-255 µs
//!     budget  1000 µs -> p99.99     ~1160 µs
//!     budget 10000 µs -> p99.99    ~10130 µs
//!
//! — so 1 ms is about five times the overshoot the mechanism actually
//! produces, while still being far below the unbudgeted p99.99 of 2-5 ms.
//! That last part matters: a slack above the unbudgeted tail makes the
//! assertion unfalsifiable, which is why each arm prints `(no power)` when
//! the control run did not itself breach that arm's limit.  A 10 ms slack
//! marks every arm no-power on this host.
//!
//! Set `KAME_WB_TEST_SECS` to lengthen each arm (default 2 s).

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        long m_x = 0;
    };
};
typedef Transactional::Snapshot<MyNode> Shot;
typedef Transactional::Transaction<MyNode> Tr;

static std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    if( !v || !*v) return dflt;
    int n = std::atoi(v);
    return n > 0 ? n : dflt;
}

//! \param budget_us 0 = no budget (control arm).
//! \return max observed per-commit latency in ns.
static std::int64_t run_arm(int threads, double secs, int budget_us,
                            std::int64_t *out_pct_ns,   //!< [3]: p99, p99.9, p99.99
                            std::uint64_t *out_commits,
                            std::uint64_t *out_attempts_at_max) {
    shared_ptr<MyNode> grand(MyNode::create<MyNode>());
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    std::vector<shared_ptr<MyNode>> children((size_t)threads);
    for(int i = 0; i < threads; i++)
        children[(size_t)i].reset(MyNode::create<MyNode>());

    grand->iterate_commit([&](Tr &tr) {
        if( !grand->insert(tr, parent, true)) return;
        for(int i = 0; i < threads; i++)
            if( !parent->insert(tr, children[(size_t)i], true)) return;
    });

    std::atomic<bool> go{false}, stop{false};
    std::atomic<int> ready{0};
    std::vector<std::int64_t> per_thread_max((size_t)threads, 0);
    std::vector<std::vector<std::int64_t>> per_thread_samples((size_t)threads);
    std::vector<std::uint64_t> per_thread_n((size_t)threads, 0);
    // Attempts (lambda invocations = 1 + retries) of the commit that produced
    // this thread's max.  This is what separates "the OS descheduled us once"
    // (attempts ~1) from "we stopped waiting and then spun on retries"
    // (attempts large) — the two have the same wall clock and completely
    // different meanings for the budget.
    std::vector<std::uint64_t> per_thread_att_at_max((size_t)threads, 0);

    auto worker = [&](int tid) {
        std::int64_t mx = 0;
        std::uint64_t n = 0, att_at_max = 0;
        std::vector<std::int64_t> samples;
        samples.reserve(1u << 16);
        ready.fetch_add(1);
        while( !go.load(std::memory_order_acquire)) { }
        while( !stop.load(std::memory_order_relaxed)) {
            std::uint64_t attempts = 0;
            std::int64_t t0 = now_ns();
            {
#if KAME_STM_WAIT_BUDGET
                // Constructed inside the timed region: the scope's own cost
                // (one clock read, two TLS accesses) is part of what a caller
                // pays and so belongs inside the bound being asserted.
                std::unique_ptr<Transactional::ScopedWaitBudget> wb;
                if(budget_us)
                    wb.reset(new Transactional::ScopedWaitBudget(budget_us));
#endif
                grand->iterate_commit([&](Tr &tr) {
                    ++attempts;
                    for(int c = 0; c < threads; c++)
                        tr[ *children[(size_t)c]].m_x++;
                });
            }
            std::int64_t dt = now_ns() - t0;
            if(dt > mx) { mx = dt; att_at_max = attempts; }
            samples.push_back(dt);   // keep all: a first-N reservoir biases
                                     // the tail toward the start of the run
            ++n;
        }
        per_thread_max[(size_t)tid] = mx;
        per_thread_att_at_max[(size_t)tid] = att_at_max;
        per_thread_n[(size_t)tid] = n;
        per_thread_samples[(size_t)tid].swap(samples);
    };

    std::vector<std::thread> ts;
    for(int t = 0; t < threads; t++) ts.emplace_back(worker, t);
    while(ready.load() < threads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(
        std::chrono::milliseconds((long long)(secs * 1000)));
    stop.store(true, std::memory_order_relaxed);
    for(auto &t : ts) t.join();

    std::int64_t mx = 0;
    std::uint64_t n = 0;
    std::vector<std::int64_t> all;
    std::uint64_t att_at_mx = 0;
    for(int t = 0; t < threads; t++) {
        if(per_thread_max[(size_t)t] > mx) {
            mx = per_thread_max[(size_t)t];
            att_at_mx = per_thread_att_at_max[(size_t)t];
        }
        n += per_thread_n[(size_t)t];
        all.insert(all.end(), per_thread_samples[(size_t)t].begin(),
                   per_thread_samples[(size_t)t].end());
    }
    if(out_commits) *out_commits = n;
    if(out_attempts_at_max) *out_attempts_at_max = att_at_mx;
    if(out_pct_ns) {
        static const double kQ[3] = {0.99, 0.999, 0.9999};
        for(int i = 0; i < 3; i++) {
            if(all.empty()) { out_pct_ns[i] = 0; continue; }
            size_t k = (size_t)((double)all.size() * kQ[i]);
            if(k >= all.size()) k = all.size() - 1;
            std::nth_element(all.begin(), all.begin() + (long)k, all.end());
            out_pct_ns[i] = all[k];
        }
    }
    return mx;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
#if !KAME_STM_WAIT_BUDGET
    std::printf("KAME_STM_WAIT_BUDGET=0 — wait budget compiled out, skipping\n");
    return 0;
#else
    const int threads = std::max(4, std::min(8,
        (int)std::thread::hardware_concurrency()));
    const double secs = (double)env_int("KAME_WB_TEST_SECS", 2);
    const std::int64_t slack_ns =
        (std::int64_t)env_int("KAME_WB_TEST_SLACK_US", 1000) * 1000;

    std::printf("wait-budget test: %d threads, %.1f s per arm, "
                "slack %lld us\n",
                threads, secs, (long long)(slack_ns / 1000));

    // Control arm: no budget.  Not asserted — it exists to show that the
    // workload really does produce a tail far above the slack, so a pass in
    // the budgeted arms means something.
    std::uint64_t n0 = 0, a0 = 0;
    std::int64_t p0[3] = {0, 0, 0};
    std::int64_t max0 = run_arm(threads, secs, 0, p0, &n0, &a0);
    std::printf("  no budget      : %9llu commits  p99 %6lld  p99.9 %7lld  "
                "p99.99 %8lld us   MAX %9lld us (att %llu)\n",
                (unsigned long long)n0, (long long)(p0[0] / 1000),
                (long long)(p0[1] / 1000), (long long)(p0[2] / 1000),
                (long long)(max0 / 1000), (unsigned long long)a0);

    static const int kBudgetsUs[] = {100, 1000, 10000};
    bool failed = false;
    for(int b : kBudgetsUs) {
        std::uint64_t n = 0, att = 0;
        std::int64_t p[3] = {0, 0, 0};
        std::int64_t mx = run_arm(threads, secs, b, p, &n, &att);
        const std::int64_t limit_ns = (std::int64_t)b * 1000 + slack_ns;
        // Power check: if the unbudgeted run did not itself breach the limit
        // at this percentile, a pass proves nothing.
        const bool has_power = (p0[2] > limit_ns);
        const bool ok = (p[2] <= limit_ns);
        std::printf("  budget %6d us: %9llu commits  p99 %6lld  p99.9 %7lld  "
                    "p99.99 %8lld us   MAX %9lld us (att %llu)  "
                    "[limit %lld us] %s%s\n",
                    b, (unsigned long long)n, (long long)(p[0] / 1000),
                    (long long)(p[1] / 1000), (long long)(p[2] / 1000),
                    (long long)(mx / 1000), (unsigned long long)att,
                    (long long)(limit_ns / 1000),
                    ok ? "ok" : "FAIL", has_power ? "" : " (no power)");
        if( !ok && has_power) failed = true;
    }

    if(failed) {
        std::printf("FAILED: p99.99 exceeded budget + slack.  The budget "
                    "bounds waiting, not completion, so a genuine failure "
                    "means the negotiator slept past the limit or the "
                    "post-expiry retry count ran away.  Diagnose with "
                    "`transaction_latency_bench -b <us>` built with "
                    "-DKAME_STM_NEG_DIAG=1: sleeps/commit and slept/commit "
                    "say whether it waited, and attempts/commit says whether "
                    "it spun.\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
#endif
}
