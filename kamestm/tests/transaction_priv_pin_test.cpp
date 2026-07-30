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
//! Keeps the 2026-07-30 field crash shape (SIGABRT via the negotiation HANG
//! watchdog while operating T1Mode during an NMR measurement) as a behavioural
//! regression net: **no thread may ever be pinned for a watchdog-class
//! stretch**.
//!
//! The field chain, reconstructed from the crash report:
//!   * an ms-scale analysis transaction (NMR pulse analyzer / T1) is
//!     re-invalidated every attempt by a burst of fresh first-attempt commits
//!     (opening/operating a form fires dozens of connector commits) — fresh
//!     transactions never enter negotiation, so privilege cannot stop them;
//!   * on the DSO thread the same loop runs under the 20 ms downstream wait
//!     budget, whose expiry deliberately bypasses fair-mode — a second
//!     fair-mode-immune contender;
//!   * the true blocker was an OWNERLESS Reserved stamp orphaned by a
//!     starvation throw during Snapshot/Transaction construction — every
//!     third-party thread that does negotiate (Thamway fetchStatus,
//!     tempcontrol, motor in the report) waited behind a ghost for
//!     3 x 5 s -> abort().  Fixed at the source (ctor exception safety).
//!
//! Roles here: D = budgeted ms-analysis on devA (the pulse analyzer),
//! U = fresh-commit burst on devA's leaf (the connector storm), M =
//! UI_DEFERRABLE ms-transactions at root scope (the T1 side, keeps the root
//! bundled), T = the third-party NORMAL driver on devB whose progress is the
//! assertion.  T must never stall for KAME_PIN_STALL_SECS (default 12 —
//! just below the 15 s watchdog).
//!
//! What this asserts, after the tier ruling (NORMAL/HIGHEST privilege never
//! expires — it is the completion guarantee): a live NORMAL holder MAY
//! legitimately shield for multi-second stretches while it retries, so the
//! stall threshold is 12 s — just under the field watchdog's 15 s — and what
//! must never happen is the GHOST-class eternal pin (an ownerless stamp),
//! which the constructor exception-safety fix guarantees against.  A briefly
//! shipped 5 s threshold encoded the rejected expiry design and was flaky by
//! construction.

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        long m_x = 0;
    };
};
typedef Transactional::Transaction<MyNode> Tr;

static long env_long(const char *name, long defv) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atol(v) : defv;
}
//! CPU-burning "analysis" of \a us microseconds (no sleeping: the field
//! analysis computes, it does not wait).
static void busy_us(long us) {
    auto t0 = std::chrono::steady_clock::now();
    while(std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t0).count() < us)
        ;
}

int main() {
    const long secs       = env_long("KAME_PIN_SECS", 25);
    const long stall_secs = env_long("KAME_PIN_STALL_SECS", 12);
    const long analyze_us = env_long("KAME_PIN_ANALYZE_US", 2000);
    std::printf("privilege-pin repro: %lds, stall>%lds fails, "
                "analysis %ldus/attempt\n", secs, stall_secs, analyze_us);

    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> devA(MyNode::create<MyNode>());
    shared_ptr<MyNode> devB(MyNode::create<MyNode>());
    shared_ptr<MyNode> panel(MyNode::create<MyNode>());
    root->insert(devA); root->insert(devB); root->insert(panel);
    std::vector<shared_ptr<MyNode>> leavesA, leavesB;
    for(int i = 0; i < 4; ++i) {
        shared_ptr<MyNode> a(MyNode::create<MyNode>());
        shared_ptr<MyNode> b(MyNode::create<MyNode>());
        leavesA.push_back(a); devA->insert(a);
        leavesB.push_back(b); devB->insert(b);
    }

    enum {T_ANALYZE = 0, T_BURST = 1, T_UI = 2, T_THIRD = 3, N_THREADS = 4};
    static const char *kNames[] = {"D analyze(budget)", "U burst",
                                   "M UI root", "T third-party"};
    std::vector<std::atomic<uint64_t>> progress(N_THREADS);
    for(auto &p : progress) p.store(0);
    std::atomic<bool> stop{false};
    std::vector<std::thread> ts;

    // D: the budgeted ms-analysis on devA — the pulse-analyzer shape.
    ts.emplace_back([&]{
        while( !stop.load(std::memory_order_relaxed)) {
            Transactional::ScopedWaitBudget budget((int64_t)20'000);
            devA->iterate_commit([&](Tr &tr){
                tr[ *devA].m_x++;
                for(auto &l : leavesA) tr[ *l].m_x++;
                busy_us(analyze_us);          // re-runs on every retry
            });
            progress[T_ANALYZE].fetch_add(1, std::memory_order_relaxed);
        }
    });
    // U: the connector storm — fresh first-attempt commits on devA's leaf,
    // never negotiating, invalidating D's window every few microseconds.
    ts.emplace_back([&]{
        while( !stop.load(std::memory_order_relaxed)) {
            leavesA[0]->iterate_commit([&](Tr &tr){ tr[ *leavesA[0]].m_x++; });
            progress[T_BURST].fetch_add(1, std::memory_order_relaxed);
        }
    });
    // M: the T1 side — UI_DEFERRABLE ms-transactions at root scope, keeping
    // the root bundled so everyone's commits must unbundle through it.
    ts.emplace_back([&]{
        Transactional::ScopedPriority pr(
            Transactional::Priority::UI_DEFERRABLE);
        while( !stop.load(std::memory_order_relaxed)) {
            root->iterate_commit([&](Tr &tr){
                tr[ *panel].m_x++;
                busy_us(1000);
            });
            progress[T_UI].fetch_add(1, std::memory_order_relaxed);
        }
    });
    // T: the victim — an unrelated NORMAL driver on its own subtree.  In the
    // field these were Thamway fetchStatus / tempcontrol / motor, and one of
    // them hit the 3 x 5 s watchdog and aborted the process.
    ts.emplace_back([&]{
        while( !stop.load(std::memory_order_relaxed)) {
            devB->iterate_commit([&](Tr &tr){ tr[ *leavesB[0]].m_x++; });
            progress[T_THIRD].fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });

    int failures = 0;
    std::vector<uint64_t> last(N_THREADS, 0), last_ms(N_THREADS, 0);
    const auto t0 = std::chrono::steady_clock::now();
    auto ms_now = [&]{ return (uint64_t)
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count(); };
    while((long)ms_now() < secs * 1000 && !failures) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        uint64_t now = ms_now();
        for(int t = 0; t < N_THREADS; ++t) {
            uint64_t v = progress[t].load(std::memory_order_relaxed);
            if(v != last[t]) { last[t] = v; last_ms[t] = now; }
            else if(now - last_ms[t] >= (uint64_t)stall_secs * 1000) {
                std::printf("STALL: %s frozen for %llu ms at count=%llu — the "
                            "field pin reproduced (a never-expiring privilege "
                            "holder that cannot win).\n", kNames[t],
                            (unsigned long long)(now - last_ms[t]),
                            (unsigned long long)v);
                ++failures;
            }
        }
    }
    stop.store(true);
    for(auto &t : ts) t.join();
    double el = ms_now() / 1000.0;
    for(int t = 0; t < N_THREADS; ++t)
        std::printf("  %-20s %12llu commits (%.0f /s)\n", kNames[t],
                    (unsigned long long)progress[t].load(),
                    progress[t].load() / el);
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
