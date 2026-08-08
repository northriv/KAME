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
//! Reading the OS arm's results (measured 2026-08, 4-CPU x86-64, non-RT
//! kernel, so treat the numbers as shape rather than as a bound):
//!
//!   * The clean 2x2, on a PREEMPT_RT host (i5-7500, isolcpus=2,3), acq
//!     commits/s: neither 146.9k; FIFO+pin only 155.0k; XSUBTREE only 89.4k;
//!     both 57.8k.  So **FIFO+pin costs nothing** (+6 %), the cross-subtree
//!     role costs **1.64x** on its own — that is the role's whole point,
//!     since without it the NORMAL peers only ever touch their own subtree and
//!     cannot contend with the acquiring driver at all — and the two together
//!     cost **2.54x**, well past the 1.54x their product predicts.  Something
//!     is super-additive.
//!   * That "something" is NOT the isolation.  Same knobs, same two-CPU shape,
//!     only the acquisition core changed: `taskset -c 0,1` (both housekeeping)
//!     gave 49.8k, `taskset -c 0,3` (onto the `nohz_full` isolated core) gave
//!     **53.9k** — the isolated core is marginally *faster*, so the
//!     wake-a-tickless-core hypothesis is refuted.  What is left is ordinary
//!     SMP: pinning forces the cross-subtree contention to be cross-core on
//!     every single conflict, where an unpinned CFS may co-locate the two
//!     threads and settle some conflicts in one cache.  Nothing RT-specific,
//!     and nothing that argues against isolating the acquisition core.
//!   * Note in passing that cramming the three housekeeping threads onto one
//!     core made them *collectively faster* (1.22M vs 911k commits/s spread
//!     over four), which is the same coherence effect seen from the other
//!     side.
//!   * Starving that same NORMAL peer did NOT pin the acquisition thread —
//!     acquisition sped back up, because a contender that is not running is
//!     not contending.  The never-expiring-privilege pin was NOT reproduced
//!     this way, and probably cannot be: privilege claims are probe-gated, so
//!     a starved thread is overwhelmingly likely to be starved while holding
//!     nothing.  `transaction_priv_expiry_test` stays the deterministic
//!     instrument for that; this arm reaches the deployment *shape*, not that
//!     specific interleaving.
//!   * `KAME_MIX_OS_STARVE=2` with the default two spinners drives the
//!     housekeeping threads to **zero** commits and trips the stall detector.
//!     Read that as an over-harsh configuration, not as an STM livelock: a
//!     count of 0 means the thread never completed even one transaction, i.e.
//!     it never got the CPU, which is what three SCHED_IDLE threads sharing
//!     one core with two spinners buys.  Use `KAME_MIX_OS_LOAD=1` or
//!     `KAME_MIX_OS_STARVE=1` to keep the holder schedulable enough to be
//!     interesting.
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
//!   KAME_MIX_LEAVES          leaves per subtree, default 4.  The field root
//!                            has ~10^3 nodes, so a root bundle costs ms and
//!                            the invalidation window for a wide UI Tx is
//!                            enormous — a 16-node tree cannot reproduce that.
//!                            Set 64-256 to model a real measurement tree
//!   KAME_MIX_OS_FIFO         >0 = acquisition thread at SCHED_FIFO of that
//!                            priority (Linux; skipped with a notice when not
//!                            permitted).  The rest stay SCHED_OTHER
//!   KAME_MIX_OS_PIN          1 = acquisition alone on the last CPU, everyone
//!                            else on CPU 0 — the shape isolcpus produces
//!   KAME_MIX_OS_STARVE       1 = UI+SCRIPTING to SCHED_IDLE, 2 = NORMAL too
//!   KAME_MIX_OS_LOAD         SCHED_OTHER spinners on the housekeeping CPU at
//!                            starve level 2, default 2 (see the caveat below)
//!   KAME_MIX_NORMAL_XSUBTREE 1 = every 4th NORMAL Tx spans the acquiring
//!                            driver's subtree (XSecondaryDriver's shape)
//!   KAME_MIX_ACQ_NORMAL      1 = acquisition thread runs at NORMAL instead of
//!                            HIGHEST: the control arm that attributes any
//!                            stall to the HIGHEST-ification or acquits it

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include "latency_hist.h"
#ifndef DISABLE_POOL_ALLOCATOR
#  include "kame_pool.h"
#endif
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cstring>
#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  include <fcntl.h>
#endif

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

// ---------------------------------------------------------------- OS class
//! The dimension this test did not have.  Every thread here has always been
//! SCHED_OTHER, so CFS runs all of them regularly and the field's rare
//! livelock stays rare: whoever holds a privilege is always scheduled soon
//! enough to finish it.
//!
//! That is not what an RT deployment looks like.  `AcquisitionPriority` keeps
//! only the OS elevation now that STM-HIGHEST is retired, so on such a host
//! the acquisition thread is `SCHED_FIFO` on an isolated core while the UI and
//! scripting threads share a loaded housekeeping core.  And the STM
//! deliberately does not rescue that: NORMAL privilege never expires (it *is*
//! the completion guarantee, and the TLA+ liveness argument assumes it
//! persists until its holder finishes), and the wait behind a live privilege
//! is exempt from the wait budget.  The bound is therefore the holder's
//! scheduling delay and nothing else — which makes it a *configuration*
//! property of the deployment, not a property of the STM.
//!
//! These knobs make that configuration reachable so the consequence can be
//! observed rather than argued about.  A stall is still the verdict.
//!
//! Nothing here ever puts a spinning thread on SCHED_FIFO: equal-priority FIFO
//! threads do not preempt one another, and the load arm would wedge the box
//! rather than starve a holder.  Starvation is modelled with SCHED_IDLE, which
//! cannot.
static bool os_set_policy(int policy, int prio) noexcept {
#if defined(__linux__)
    sched_param sp;
    std::memset( &sp, 0, sizeof(sp));
    sp.sched_priority = prio;
    return pthread_setschedparam(pthread_self(), policy, &sp) == 0;
#else
    (void)policy; (void)prio;
    return false;
#endif
}
//! Explicit, at the top of every thread body.  pthread_create defaults to
//! PTHREAD_INHERIT_SCHED, so a thread spawned after another has elevated
//! itself would silently come up elevated too — the bug this project has
//! already paid for once, in bench_rt_wcet.
static void os_be_ordinary() noexcept {
#if defined(__linux__)
    os_set_policy(SCHED_OTHER, 0);
#endif
}
static void os_be_starved() noexcept {
#if defined(__linux__)
    if( !os_set_policy(SCHED_IDLE, 0)) os_be_ordinary();
#endif
}
static void os_pin(long cpu) noexcept {
#if defined(__linux__)
    if(cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO( &set);
    CPU_SET((int)cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
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
    const long n_leaves    = env_long("KAME_MIX_LEAVES", 4);
    const bool acq_normal  = env_long("KAME_MIX_ACQ_NORMAL", 0) != 0;
    const long os_fifo     = env_long("KAME_MIX_OS_FIFO", 0);
    //! 0 = off, 1 = the lowprio tiers (UI + SCRIPTING) go SCHED_IDLE,
    //! 2 = the NORMAL peers as well.  The two levels ask different questions.
    //! Expiry is a lowprio-only mechanism, so level 1 starves holders whose
    //! privilege the STM *can* revoke and therefore tests that rescue path.
    //! Level 2 starves a NORMAL holder, whose privilege never expires and
    //! behind which the wait budget is exempt — there the only bound left is
    //! the OS scheduler, which is the case worth quantifying before choosing
    //! a policy for raiseAcquisitionOSPriority_().
    const long os_starve   = env_long("KAME_MIX_OS_STARVE", 0);
    //! 1 = every 4th NORMAL transaction spans the acquiring driver's subtree
    //! (the XSecondaryDriver / ms-analysis role).  Off by default so the
    //! existing arms are unchanged.
    const bool normal_xsub = env_long("KAME_MIX_NORMAL_XSUBTREE", 0) != 0;
    //! >0 turns the record-commit distribution into an assertion: any record
    //! commit longer than this many microseconds fails the run.  0 (default)
    //! reports the distribution and asserts nothing, because an absolute
    //! latency is a property of the host, not of the STM — the same split
    //! bench_rt_wcet makes between its machine-independent violation count and
    //! its machine-specific histogram.
    const long deadline_us = env_long("KAME_MIX_DEADLINE_US", 0);
    //! >0 arms a breaktrace: the FIRST record commit longer than this many
    //! microseconds writes a marker into ftrace and switches tracing off, so
    //! the buffer freezes holding whatever ran just before it.  The same
    //! instrument cyclictest's --breaktrace is, aimed at a commit instead of a
    //! timer wake-up — because hunting a rare fixed-cost event by reading a
    //! running trace is hopeless, while catching it in the act is routine.
    //! Needs tracefs mounted and root; says so and stays disarmed otherwise.
    const long trace_us = env_long("KAME_MIX_TRACE_US", 0);
    //! Samples in the first this-many milliseconds are counted but kept OUT of
    //! the histogram and cannot fire the breaktrace.  Without it the largest
    //! sample of every run is a cold-start artefact and the breaktrace can
    //! never catch anything else — the 2026-08 RT investigation had to disable
    //! the allocator's pre-fill entirely to get past it.
    const long warmup_ms = env_long("KAME_MIX_WARMUP_MS", 500);
    //! Precondition 2 of the realtime contract: prewarm from the realtime
    //! thread, before the time-critical section.  On by default because the
    //! contract requires it and because this test previously did not do it —
    //! which is exactly how it came to measure a 400 us first-commit spike
    //! (the pool's per-slot next-pointer pre-fill faulting 5 size classes x 64
    //! pages) and mistake it for a recurring event.  Set 0 to reproduce that.
    const bool do_prewarm  = env_long("KAME_MIX_PREWARM", 1) != 0;
    const bool os_pin_on   = env_long("KAME_MIX_OS_PIN", 0) != 0;
    std::printf("mixed-priority livelock hunt: %lds, stall>%lds fails, "
                "acq=%s duty %ldus, UI period %ldus, +%ld NORMAL, "
                "+%ld SCRIPTING, %ld leaves/subtree\n",
                secs, stall_secs, acq_normal ? "NORMAL(control)" : "HIGHEST",
                hi_duty_us, ui_period_us, n_normals, n_scripting, n_leaves);

    // Probe the OS arm up front so an unprivileged run says so instead of
    // reporting a green RT result it never ran.
    //! Derive the CPUs from this process's AFFINITY MASK, not from the online
    //! count: `taskset -c 0,1 ./test` then chooses which cores the arm uses,
    //! which is the discriminator for "is the penalty about crossing cores, or
    //! about crossing onto a nohz_full one" — no extra knob needed.
    std::vector<long> cpus;
#if defined(__linux__)
    {
        cpu_set_t set;
        CPU_ZERO( &set);
        if(sched_getaffinity(0, sizeof(set), &set) == 0) {
            for(int c = 0; c < CPU_SETSIZE; ++c)
                if(CPU_ISSET(c, &set)) cpus.push_back(c);
        }
    }
#endif
    if(cpus.empty()) cpus.push_back(0);
    const long ncpu = (long)cpus.size();
    bool fifo_ok = false;
    if(os_fifo > 0) {
        fifo_ok = os_set_policy(SCHED_FIFO, (int)os_fifo);
        os_be_ordinary();               // probe only; the thread sets its own
        if( !fifo_ok)
            std::printf("  NOTE: SCHED_FIFO %ld was requested but is not "
                        "permitted (need CAP_SYS_NICE or RLIMIT_RTPRIO) — the "
                        "OS arm is SKIPPED, this run is SCHED_OTHER "
                        "throughout.\n", os_fifo);
    }
    if(os_pin_on && (ncpu < 2)) {
        std::printf("  NOTE: KAME_MIX_OS_PIN needs >= 2 usable CPUs (have "
                    "%ld) — pinning SKIPPED.\n", ncpu);
    }
    const bool pin_ok = os_pin_on && (ncpu >= 2);
    //! Acquisition alone on the LAST allowed CPU, everyone else on the first:
    //! the shape `isolcpus` produces, without needing the kernel parameter.
    const long cpu_acq   = pin_ok ? cpus.back()  : -1;
    const long cpu_house = pin_ok ? cpus.front() : -1;
    if(os_fifo > 0 || os_starve || pin_ok)
        std::printf("  OS arm: fifo=%s starve(SCHED_IDLE lowprio)=%s "
                    "pin=%s (acq->cpu%ld, others->cpu%ld), %ld CPUs\n",
                    fifo_ok ? "yes" : "no",
                    (os_starve >= 2) ? "lowprio+NORMAL" :
                        (os_starve ? "lowprio" : "no"),
                    pin_ok ? "yes" : "no", cpu_acq, cpu_house, ncpu);

    // The measurement tree: root -> {devA, devB, panel}, four leaves each.
    // devA is the acquiring driver's subtree; entriesA models its scalar
    // entries, written by the demoted downstream.
    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> devA(MyNode::create<MyNode>());
    shared_ptr<MyNode> devB(MyNode::create<MyNode>());
    shared_ptr<MyNode> panel(MyNode::create<MyNode>());
    root->insert(devA); root->insert(devB); root->insert(panel);
    std::vector<shared_ptr<MyNode>> leavesA, leavesB, leavesP;
    for(long i = 0; i < n_leaves; ++i) {
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
    //! Written only by the acquisition thread, read only after join().
    Hist acq_hist;
    acq_hist.reset();
    std::atomic<uint64_t> cold_n{0};   //!< commits dropped as warm-up

    // Breaktrace plumbing.  Both descriptors are opened up front so the hot
    // path only ever does two write()s, and only once.
    int trace_marker_fd = -1, tracing_on_fd = -1;
    std::atomic<bool> trace_fired{false};
#if defined(__linux__)
    if(trace_us > 0) {
        static const char *kRoots[] = {"/sys/kernel/tracing",
                                       "/sys/kernel/debug/tracing"};
        for(const char *r : kRoots) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s/trace_marker", r);
            trace_marker_fd = ::open(buf, O_WRONLY | O_CLOEXEC);
            if(trace_marker_fd < 0) continue;
            std::snprintf(buf, sizeof(buf), "%s/tracing_on", r);
            tracing_on_fd = ::open(buf, O_WRONLY | O_CLOEXEC);
            if(tracing_on_fd >= 0) {
                std::printf("  breaktrace armed at %ld us via %s\n",
                            trace_us, r);
                break;
            }
            ::close(trace_marker_fd);
            trace_marker_fd = -1;
        }
        if(tracing_on_fd < 0)
            std::printf("  NOTE: KAME_MIX_TRACE_US needs tracefs and root — "
                        "breaktrace DISARMED (mount it and re-run as root; "
                        "the latency histogram below is unaffected).\n");
    }
#endif

    // --- The acquisition thread, oscillating exactly like finishWritingRaw:
    // record commit at HIGHEST, then the demoted downstream at NORMAL under
    // the 20 ms budget, every cycle.
    ts.emplace_back([&]{
        if(fifo_ok) os_set_policy(SCHED_FIFO, (int)os_fifo);
        else        os_be_ordinary();
        os_pin(cpu_acq);
#ifndef DISABLE_POOL_ALLOCATOR
        if(do_prewarm) {
            //! Cover the small classes the STM's Payload clones land in.
            //! Over-covering is free; missing one puts its first chunk claim
            //! back on the measured path.
            static const std::size_t kSizes[] =
                {16, 32, 48, 64, 96, 128, 192, 256, 512, 1024};
            unsigned counts[sizeof(kSizes) / sizeof(kSizes[0])];
            for(auto &c : counts) c = 64u;
            if(kame_pool_prewarm(kSizes, counts,
                                 (unsigned)(sizeof(kSizes) / sizeof(kSizes[0]))))
                std::printf("  NOTE: kame_pool_prewarm did not fit — the first "
                            "commits will show cold-path outliers.\n");
        }
#endif
        const std::uint64_t t_warm_end = now_ns() +
            (std::uint64_t)warmup_ms * 1000000ull;
        Transactional::ScopedPriority pr(acq_normal
            ? Transactional::Priority::NORMAL
            : Transactional::Priority::HIGHEST);
        while( !stop.load(std::memory_order_relaxed)) {
            {   // the record commit (multi-nodal, driver scope).
                //! Timed, because "the acquisition thread kept up on average"
                //! and "no record took longer than X" are different claims and
                //! only the second one is a realtime one.  The counters below
                //! answer the first; this histogram answers the second.
                Transactional::ScopedWaitBudget budget((int64_t)20'000);
                const std::uint64_t t_rec = now_ns();
                devA->iterate_commit([&](Tr &tr){
                    tr[ *devA].m_x++;
                    for(auto &l : leavesA) tr[ *l].m_x++;
                });
                const std::uint64_t t_end = now_ns();
                const std::uint64_t dt_rec = t_end - t_rec;
                const bool warm = (t_end >= t_warm_end);
                if(warm) acq_hist.add(dt_rec);
                else     cold_n.fetch_add(1, std::memory_order_relaxed);
#if defined(__linux__)
                if(warm && (tracing_on_fd >= 0) &&
                   (dt_rec >= (std::uint64_t)trace_us * 1000ull) &&
                   !trace_fired.exchange(true, std::memory_order_relaxed)) {
                    char m[96];
                    int n = std::snprintf(m, sizeof(m),
                        "KAME_MIX: record commit took %llu ns\n",
                        (unsigned long long)dt_rec);
                    ssize_t w = ::write(trace_marker_fd, m, (size_t)n);
                    w = ::write(tracing_on_fd, "0\n", 2);   // freeze the buffer
                    (void)w;
                }
#endif
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
        if(os_starve >= 1) os_be_starved(); else os_be_ordinary();
        os_pin(cpu_house);
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
            leavesP[i % leavesP.size()]->iterate_commit([&](Tr &tr){
                tr[ *leavesP[i % leavesP.size()]].m_x++;
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
            if(os_starve >= 1) os_be_starved(); else os_be_ordinary();
            os_pin(cpu_house);
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
                        tr[ *leavesP[(i / 16 + (uint64_t)k) % leavesP.size()]].m_x++;
                    });
                progress[T_SCRIPT0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- NORMAL: other drivers on their own subtree.
    for(long k = 0; k < n_normals; ++k) {
        ts.emplace_back([&, k]{
            //! Only at starve level 2.  A starved NORMAL holder is the
            //! unbounded case — its privilege never expires and the wait
            //! behind it is budget-exempt — so it is kept behind its own
            //! level rather than riding along with the revocable tiers.
            if(os_starve >= 2) os_be_starved(); else os_be_ordinary();
            os_pin(cpu_house);
            Transactional::ScopedPriority pr(Transactional::Priority::NORMAL);
            uint64_t i = 0;
            while( !stop.load(std::memory_order_relaxed)) {
                ++i;
                if(normal_xsub && ((i % 4) == 0)) {
                    //! XSecondaryDriver's shape: a NORMAL transaction whose
                    //! scope SPANS the acquiring driver's subtree, because it
                    //! reads the primary's record and writes its own result.
                    //! Without this role the NORMAL peers only ever touch
                    //! devB, so they can never hold privilege on the linkage
                    //! the acquisition thread commits to — and the
                    //! never-expiring, budget-exempt case cannot arise no
                    //! matter how hard they are starved.  This is the role in
                    //! the 2026-07-30 field crash that transaction_priv_expiry
                    //! _test reproduces white-box.
                    root->iterate_commit([&](Tr &tr){
                        (void)tr[ *devA].m_x;
                        tr[ *leavesB[(i + (uint64_t)k) % leavesB.size()]].m_x++;
                    });
                }
                else {
                    devB->iterate_commit([&](Tr &tr){
                        tr[ *leavesB[(i + (uint64_t)k) % leavesB.size()]].m_x++;
                    });
                }
                progress[T_NORMAL0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- Housekeeping-core load.  SCHED_IDLE only starves a thread when
    // something SCHED_OTHER wants the same CPU; at level 1 the NORMAL peers
    // supply that, but at level 2 they are idled too and every thread on the
    // housekeeping CPU would run freely again.  These spinners are what makes
    // "the housekeeping core is saturated" true.  SCHED_OTHER, never FIFO, and
    // pinned away from the acquisition CPU.
    const long n_load = (os_starve >= 2) ? env_long("KAME_MIX_OS_LOAD", 2) : 0;
    for(long k = 0; k < n_load; ++k) {
        ts.emplace_back([&]{
            os_be_ordinary();
            os_pin(cpu_house);
            volatile uint64_t sink = 0;
            while( !stop.load(std::memory_order_relaxed))
                for(int i = 0; i < 4096; ++i) sink = sink + i;
            (void)sink;
        });
    }
    if(n_load)
        std::printf("  OS arm: +%ld SCHED_OTHER spinner(s) on cpu%ld\n",
                    n_load, cpu_house);

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

    // The realtime question: not "did it keep up" but "did any one record take
    // too long".  Percentiles are printed only where the sample count can
    // support them.
    std::printf("  acq record-commit latency  (warm; %llu cold commit(s) in the "
                "first %ld ms dropped)\n", (unsigned long long)cold_n.load(),
                warmup_ms);
    std::printf("    n=%llu  mean=%llu ns  p50=%llu",
                (unsigned long long)acq_hist.n,
                (unsigned long long)(acq_hist.n ? acq_hist.sum / acq_hist.n : 0),
                (unsigned long long)acq_hist.pct(0.50));
    static const double kP[] = {0.99, 0.999, 0.9999, 0.99999};
    static const char *kPN[] = {"p99", "p99.9", "p99.99", "p99.999"};
    for(int i = 0; i < 4; ++i)
        if(acq_hist.supports(kP[i]))
            std::printf(" %s=%llu", kPN[i],
                        (unsigned long long)acq_hist.pct(kP[i]));
    std::printf("  MAX=%llu ns\n", (unsigned long long)acq_hist.max);
    if(deadline_us > 0) {
        const std::uint64_t over =
            acq_hist.at_or_above((std::uint64_t)deadline_us * 1000ull);
        std::printf("  over the %ld us deadline: %llu of %llu\n",
                    deadline_us, (unsigned long long)over,
                    (unsigned long long)acq_hist.n);
        if(over) ++failures;
    }
    else {
        std::printf("  (no deadline asserted; set KAME_MIX_DEADLINE_US to make "
                    "the MAX above a pass/fail — and quote it against the "
                    "host's own floor, e.g. cyclictest max)\n");
    }
#if defined(__linux__)
    if(tracing_on_fd >= 0) {
        std::printf(trace_fired.load()
            ? "  breaktrace FIRED — tracing is off and the buffer holds the "
              "run-up; read it with `cat /sys/kernel/tracing/trace`, then "
              "`echo 1 > .../tracing_on` to re-arm.\n"
            : "  breaktrace did not fire (no commit reached %ld us).\n",
            trace_us);
        ::close(tracing_on_fd);
        ::close(trace_marker_fd);
    }
#endif
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
