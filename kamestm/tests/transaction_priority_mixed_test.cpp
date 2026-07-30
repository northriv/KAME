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
//! HIGHEST + NORMAL + UI_DEFERRABLE mixed-priority livelock hunt.
//!
//! Motivated by a field report: KAME livelocks rarely when the UI is operated
//! during an NMR measurement, suspicion on the HIGHEST-ification.  This test
//! reproduces that deployment's ROLES, not just its thread counts:
//!
//!   acquisition  ONE thread oscillating exactly like finishWritingRaw: the
//!                record commit on its driver subtree at HIGHEST, then
//!                ScopedDemoteRealtime + the 20 ms ScopedWaitBudget for the
//!                demoted downstream (entry writes on the same subtree and a
//!                visualize-ish snapshot) at NORMAL, then back.  The
//!                HIGHEST<->NORMAL oscillation on one thread is the real
//!                deployment shape — a first draft used a separate
//!                always-NORMAL downstream thread, which is both harsher and
//!                wrong (it never self-throttles the HIGHEST churn).
//!   NORMAL       other drivers' threads on their own subtree.
//!   UI           UI_DEFERRABLE, the main-thread mix: frequent ROOT Snapshots
//!                (graph redraws bundle the root and absorb the driver
//!                packets — the documented always-fail shape for descendant
//!                commits), leaf widget writes, occasional root-scope
//!                transactions, structural insert/release churn (tool
//!                creation), and — the typical NMR trigger — writes into the
//!                MEASURING driver's own subtree (changing averaging etc.
//!                mid-acquisition).
//!
//! Livelock is detected as STALL: a per-thread commit counter that stops
//! advancing for KAME_MIX_STALL_SECS (default 5) while wall time advances.
//! No starvation handler is installed (kamestm default), so a livelock
//! manifests as no-progress rather than an exception — which is what the
//! field sees, since a stuck UI iterate_commit never returns to the event
//! loop.
//!
//! Knobs (env):
//!   KAME_MIX_SECS            run length, default 10 (ctest); set 60+ to soak
//!   KAME_MIX_STALL_SECS      stall threshold, default 5
//!   KAME_MIX_HIGHEST_DUTY_US pause between records, default 0 = flat out
//!   KAME_MIX_UI_PERIOD_US    pause between UI actions, default 0 = flat out
//!   KAME_MIX_NORMALS         extra NORMAL driver threads, default 1
//!   KAME_MIX_SCRIPTING       SCRIPTING threads, default 1 — the field has TWO
//!                            lowprio threads (main UI_DEFERRABLE + the Python
//!                            thread), and the bench already knows lowprio
//!                            threads starve each *other* at 2+, not at 1
//!   KAME_MIX_UI_WIDE         every Nth UI action is a root-scope Tx,
//!                            default 8; 1 = hostile (every action wide)
//!   KAME_MIX_ACQ_NORMAL      1 = acquisition thread runs at NORMAL instead of
//!                            HIGHEST: the control arm that attributes any
//!                            stall to the HIGHEST-ification or acquits it

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
typedef Transactional::Snapshot<MyNode> Ss;

static long env_long(const char *name, long defv) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atol(v) : defv;
}
static void pause_us(long us) {
    if(us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(us));
}

int main() {
    const long secs        = env_long("KAME_MIX_SECS", 10);
    const long stall_secs  = env_long("KAME_MIX_STALL_SECS", 5);
    const long hi_duty_us  = env_long("KAME_MIX_HIGHEST_DUTY_US", 0);
    const long ui_period_us= env_long("KAME_MIX_UI_PERIOD_US", 0);
    const long n_normals   = env_long("KAME_MIX_NORMALS", 1);
    const long n_scripting = env_long("KAME_MIX_SCRIPTING", 1);
    const long ui_wide     = env_long("KAME_MIX_UI_WIDE", 8);
    const bool acq_normal  = env_long("KAME_MIX_ACQ_NORMAL", 0) != 0;
    std::printf("mixed-priority livelock hunt: %lds, stall>%lds fails, "
                "acq=%s duty %ldus, UI period %ldus, +%ld NORMAL, "
                "+%ld SCRIPTING\n",
                secs, stall_secs, acq_normal ? "NORMAL(control)" : "HIGHEST",
                hi_duty_us, ui_period_us, n_normals, n_scripting);

    // The measurement tree: root -> {devA, devB, panel}, four leaves each.
    // devA is the acquiring driver's subtree; entriesA models its scalar
    // entries, written by the demoted downstream.
    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> devA(MyNode::create<MyNode>());
    shared_ptr<MyNode> devB(MyNode::create<MyNode>());
    shared_ptr<MyNode> panel(MyNode::create<MyNode>());
    root->insert(devA); root->insert(devB); root->insert(panel);
    std::vector<shared_ptr<MyNode>> leavesA, leavesB, leavesP;
    for(int i = 0; i < 4; ++i) {
        shared_ptr<MyNode> a(MyNode::create<MyNode>());
        shared_ptr<MyNode> b(MyNode::create<MyNode>());
        shared_ptr<MyNode> p(MyNode::create<MyNode>());
        leavesA.push_back(a); devA->insert(a);
        leavesB.push_back(b); devB->insert(b);
        leavesP.push_back(p); panel->insert(p);
    }

    enum {T_HIGHEST = 0, T_DOWNSTREAM = 1, T_UI = 2, T_SCRIPT0 = 3};
    const int T_NORMAL0 = T_SCRIPT0 + (int)n_scripting;
    const int nthreads = T_NORMAL0 + (int)n_normals;
    std::vector<std::atomic<uint64_t>> progress(nthreads);
    for(auto &p : progress) p.store(0);
    std::atomic<bool> stop{false};
    std::vector<std::thread> ts;

    // --- The acquisition thread, oscillating exactly like finishWritingRaw:
    // record commit at HIGHEST, then the demoted downstream at NORMAL under
    // the 20 ms budget, every cycle.
    ts.emplace_back([&]{
        Transactional::ScopedPriority pr(acq_normal
            ? Transactional::Priority::NORMAL
            : Transactional::Priority::HIGHEST);
        while( !stop.load(std::memory_order_relaxed)) {
            {   // the record commit (multi-nodal, driver scope).
                Transactional::ScopedWaitBudget budget((int64_t)20'000);
                devA->iterate_commit([&](Tr &tr){
                    tr[ *devA].m_x++;
                    for(auto &l : leavesA) tr[ *l].m_x++;
                });
                progress[T_HIGHEST].fetch_add(1, std::memory_order_relaxed);
                // the demoted downstream: entry writes + visualize snapshot.
                Transactional::ScopedDemoteRealtime _demoted;
                leavesA[0]->iterate_commit([&](Tr &tr){
                    tr[ *leavesA[0]].m_x++;
                });
                {
                    Ss shot( *devA);
                    (void)shot[ *leavesA[1]].m_x;
                }
                progress[T_DOWNSTREAM].fetch_add(1, std::memory_order_relaxed);
            }
            pause_us(hi_duty_us);
        }
    });

    // --- UI_DEFERRABLE: the main-thread mix.
    ts.emplace_back([&]{
        Transactional::ScopedPriority pr(
            Transactional::Priority::UI_DEFERRABLE);
        uint64_t i = 0;
        while( !stop.load(std::memory_order_relaxed)) {
            ++i;
            {   // graph redraw: root Snapshot — bundles the whole tree.
                Ss shot( *root);
                (void)shot[ *devA].m_x;
            }
            // widget edit: leaf write.
            leavesP[i % 4]->iterate_commit([&](Tr &tr){
                tr[ *leavesP[i % 4]].m_x++;
            });
            if(i % (uint64_t)ui_wide == 0) {
                // settings apply: root-scope transaction.
                root->iterate_commit([&](Tr &tr){
                    tr[ *panel].m_x++;
                    tr[ *devB].m_x++;
                });
            }
            if(i % 16 == 0) {
                // the classic NMR trigger: a settings write into the
                // MEASURING driver's own subtree, mid-acquisition.
                leavesA[2]->iterate_commit([&](Tr &tr){
                    tr[ *leavesA[2]].m_x++;
                });
            }
            if(i % 32 == 0) {
                // tool/driver creation & removal: structural churn.
                shared_ptr<MyNode> tmp(MyNode::create<MyNode>());
                panel->insert(tmp);
                panel->release(tmp);
            }
            progress[T_UI].fetch_add(1, std::memory_order_relaxed);
            pause_us(ui_period_us);
        }
    });

    // --- SCRIPTING: the Python thread's shape — wide snapshots (reading
    // scalar entries / node tree) plus occasional writes at driver and panel
    // scope.  The second lowprio thread the field always has.
    for(long k = 0; k < n_scripting; ++k) {
        ts.emplace_back([&, k]{
            Transactional::ScopedPriority pr(
                Transactional::Priority::SCRIPTING);
            uint64_t i = 0;
            while( !stop.load(std::memory_order_relaxed)) {
                ++i;
                {
                    Ss shot( *root);           // read the tree, like a script
                    (void)shot[ *devA].m_x;
                }
                if(i % 4 == 0)
                    leavesA[3]->iterate_commit([&](Tr &tr){
                        tr[ *leavesA[3]].m_x++;    // script pokes the driver
                    });
                if(i % 16 == 0)
                    panel->iterate_commit([&](Tr &tr){
                        tr[ *leavesP[(i / 16 + (uint64_t)k) % 4]].m_x++;
                    });
                progress[T_SCRIPT0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- NORMAL: other drivers on their own subtree.
    for(long k = 0; k < n_normals; ++k) {
        ts.emplace_back([&, k]{
            Transactional::ScopedPriority pr(Transactional::Priority::NORMAL);
            uint64_t i = 0;
            while( !stop.load(std::memory_order_relaxed)) {
                ++i;
                devB->iterate_commit([&](Tr &tr){
                    tr[ *leavesB[(i + (uint64_t)k) % 4]].m_x++;
                });
                progress[T_NORMAL0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- watchdog: stall = livelock.
    static const char *kNames[] = {"acq(record)", "  demoted downstream",
                                   "UI_DEFERRABLE", "SCRIPTING", "NORMAL"};
    auto name_of = [&](int t){
        return kNames[t < T_SCRIPT0 ? t : (t < T_NORMAL0 ? 3 : 4)]; };
    std::vector<uint64_t> last(nthreads, 0), last_change_ms(nthreads, 0);
    int failures = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto ms_now = [&]{ return (uint64_t)
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count(); };
    while((long)ms_now() < secs * 1000 && !failures) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        uint64_t now_ms = ms_now();
        for(int t = 0; t < nthreads; ++t) {
            uint64_t v = progress[t].load(std::memory_order_relaxed);
            if(v != last[t]) { last[t] = v; last_change_ms[t] = now_ms; }
            else if(now_ms - last_change_ms[t] >= (uint64_t)stall_secs * 1000) {
                std::printf("STALL: thread %d (%s) made no progress for "
                            "%llu ms at count=%llu — livelock.\n",
                            t, name_of(t),
                            (unsigned long long)(now_ms - last_change_ms[t]),
                            (unsigned long long)v);
                ++failures;
            }
        }
    }
    stop.store(true);
    for(auto &t : ts) t.join();

    double el = ms_now() / 1000.0;
    for(int t = 0; t < nthreads; ++t)
        std::printf("  %-24s %12llu commits  (%.0f /s)\n", name_of(t),
                    (unsigned long long)progress[t].load(),
                    progress[t].load() / el);
    std::printf("  priv strips (Rule 0): %llu\n",
        (unsigned long long)Transactional::detail::g_priv_strips.load());
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
