/***************************************************************************
 * transaction_payload_integrity_3level_mixed_test.cpp
 *
 * 3-level variant of the Synchrobench-style mixed-contention benchmark.
 *
 * Tree: Grand -> Parent -> {Child_0 .. Child_{N-1}} (1 leaf per thread).
 *
 * CrossRatio K knob:
 *   K=0 : pure disjoint — leaf commits only (no contention).
 *   K=1 : every tx commits at Grand — 3-level full bundle, maximum
 *         serialization across threads (7 CAS per commit per the
 *         protocol accounting in §2.3).
 *   K   : 1-in-K iterations is a Grand-scope commit; remaining K-1
 *         are disjoint leaf commits. Matches Synchrobench's update
 *         rate at discrete ratios (K=10 ≈ 10 % cross-level, K=100 ≈ 1 %).
 *
 * Terminal invariant (stress + correctness):
 *   child[i].m_x == (leaf_count_thread_i + grand_count_total) % MaxPayload
 *   where grand_count_total = sum of Grand-scope commits across all
 *   threads (each Grand commit increments every child exactly once).
 *
 * Invoke:
 *   ./transaction_payload_integrity_3level_mixed_test
 *       [StressSeconds] [NumThreads] [MaxPayload] [CrossRatio]
 *
 * Copyright (C) KAME project. GPL v2+.
 ***************************************************************************/

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        unsigned int m_x = 0;
    };
};
typedef Transactional::Snapshot<MyNode> Shot;
typedef Transactional::Transaction<MyNode> Tr;

int main(int argc, char** argv) {
    int StressSeconds = (argc > 1) ? std::atoi(argv[1]) : 0;
    int NumThreads    = (argc > 2) ? std::atoi(argv[2]) : 4;
    int MaxPayload    = (argc > 3) ? std::atoi(argv[3]) : 3;
    int CrossRatio    = (argc > 4) ? std::atoi(argv[4]) : 0;
    int MaxCommits    = (StressSeconds > 0) ? 0x7fffffff : 10000;

    if(NumThreads < 1) NumThreads = 1;
    if(MaxPayload < 1) MaxPayload = 1;
    if(CrossRatio < 0) CrossRatio = 0;

    // NUMA first-touch: when KAME_FIRSTTOUCH=1, leaves are allocated and
    // inserted from the worker thread that owns them. On NUMA systems the
    // OS scheduler typically runs each std::thread on a single socket, so
    // first-touch policy places the leaf Node memory on that socket — the
    // worker's subsequent CAS / payload-clone activity then stays
    // socket-local. Default 0 preserves the original main-thread alloc
    // pattern (used by paper figures).
    const char *ft_env = std::getenv("KAME_FIRSTTOUCH");
    const bool FirstTouch = ft_env && ft_env[0] == '1';

    fprintf(stderr, "[3level_mixed] StressSeconds=%d NumThreads=%d "
                    "MaxPayload=%d CrossRatio=%d FirstTouch=%d\n",
            StressSeconds, NumThreads, MaxPayload, CrossRatio,
            FirstTouch ? 1 : 0);

    // Tree: Grand -> Parent -> N children.
    shared_ptr<MyNode> grand(MyNode::create<MyNode>());
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    grand->insert(parent);
    std::vector<shared_ptr<MyNode>> children(NumThreads);
    if( !FirstTouch) {
        for(int i = 0; i < NumThreads; ++i) {
            children[i].reset(MyNode::create<MyNode>());
            parent->insert(children[i]);
        }
    }
    // FirstTouch path: children are created inside the worker (below).

    std::atomic<int> barrier{0};
    std::atomic<bool> warming_up{true};
    std::atomic<bool> stop{false};
    // Per-thread counts split into two: leaf_count / grand_count_total
    // capture the full life-of-test increments (needed by the terminal
    // payload-equivalence check), while leaf_count_timed /
    // grand_count_timed are reset to zero when `warming_up` flips false
    // — these feed the reported commits/s. The 1-second warmup lets
    // whatever bundle/unbundle settlement the multi-threaded workload
    // triggers finish before measurement, which matters for disjoint
    // leaf commits in the 3-level tree (observed as flat N=2 scaling
    // when no warmup is done).
    std::vector<long long> leaf_count(NumThreads, 0);
    std::vector<long long> leaf_count_timed(NumThreads, 0);
    std::atomic<long long> grand_count_total{0};
    std::atomic<long long> grand_count_timed{0};

    auto worker = [&](int tid) {
        if(FirstTouch) {
            // First-touch the leaf Node on this thread's NUMA node so all
            // subsequent CAS / payload activity stays socket-local. The
            // STM Tx wrapping `parent->insert` is lock-free and safe to
            // call from concurrent workers.
            children[tid].reset(MyNode::create<MyNode>());
            parent->insert(children[tid]);
        }
        barrier.fetch_add(1);
        while(barrier.load() < NumThreads) std::this_thread::yield();

        long long my_leaf = 0,  my_leaf_t = 0;
        long long my_grand = 0, my_grand_t = 0;
        bool was_warming = true;
        unsigned int mp = (unsigned)MaxPayload;
        shared_ptr<MyNode> &my_leaf_node = children[tid];

        for(int iter = 0; iter < MaxCommits; ++iter) {
            if(StressSeconds > 0 && stop.load(std::memory_order_relaxed)) break;
            bool warming = warming_up.load(std::memory_order_relaxed);
            if(was_warming && !warming) {
                // Just flipped to timed phase — reset the timed counter
                // baseline so the first timed iteration starts from 0.
                was_warming = false;
            }

            bool do_grand = (CrossRatio > 0) && ((iter % CrossRatio) == 0);
            if(do_grand) {
                // Grand-scope tx — 3-level bundle: bundle Parent + all
                // children up into Grand, CAS Grand, unbundle back.
                // 7 CAS for N=2; general form 1 + 2(N+1) for N children.
                grand->iterate_commit([&](Tr& tr) {
                    for(int c = 0; c < NumThreads; ++c) {
                        tr[*children[c]].m_x = (tr[*children[c]].m_x + 1) % mp;
                    }
                });
                ++my_grand;
                if(!warming) ++my_grand_t;
            } else {
                my_leaf_node->iterate_commit([&](Tr& tr) {
                    tr[*my_leaf_node].m_x = (tr[*my_leaf_node].m_x + 1) % mp;
                });
                ++my_leaf;
                if(!warming) ++my_leaf_t;
            }
        }

        leaf_count[tid] = my_leaf;
        leaf_count_timed[tid] = my_leaf_t;
        grand_count_total.fetch_add(my_grand);
        grand_count_timed.fetch_add(my_grand_t);
    };

    std::vector<std::thread> threads;
    threads.reserve(NumThreads);
    for(int i = 0; i < NumThreads; ++i) threads.emplace_back(worker, i);

    // Warmup (1s by default) — workers run the mixed workload with
    // warming_up=true so their timed counters stay at zero; the real
    // bundle/unbundle settlement happens during this phase.
    const int WarmupSeconds = (StressSeconds > 0) ? 1 : 0;
    auto t_start = std::chrono::steady_clock::now();
    if(WarmupSeconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(WarmupSeconds));
    }
    warming_up.store(false, std::memory_order_release);
    auto t_timed_start = std::chrono::steady_clock::now();

    if(StressSeconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(StressSeconds));
        stop.store(true, std::memory_order_release);
    }

    for(auto& t : threads) t.join();

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_timed_start).count();
    (void)t_start;

    // Terminal payload-equivalence uses the LIFE-of-test counters
    // (leaf_count + grand_count_total) because the warmup also
    // incremented payloads; those warmup commits must be reflected in
    // the expected value.
    long long gcount_full = grand_count_total.load();
    bool ok = true;
    for(int i = 0; i < NumThreads; ++i) {
        Shot s(*children[i]);
        unsigned int got = s[*children[i]].m_x;
        unsigned int expected = (unsigned)((leaf_count[i] + gcount_full) % MaxPayload);
        if(got != expected) {
            fprintf(stderr, "FAIL: child[%d].m_x=%u expected=%u "
                           "(leaf=%lld grand_total=%lld)\n",
                    i, got, expected, leaf_count[i], gcount_full);
            ok = false;
        }
    }

    // Reported rates use the TIMED counters only — warmup contribution
    // is excluded so the figure reflects steady-state throughput.
    long long gcount_t = grand_count_timed.load();
    long long leaf_total_t = 0;
    for(int i = 0; i < NumThreads; ++i) leaf_total_t += leaf_count_timed[i];
    long long total_commits = gcount_t * (long long)NumThreads + leaf_total_t;
    if(StressSeconds > 0) {
        printf("[3level_mixed stress %ds warmup %ds] N=%d CrossRatio=%d "
               "leaf_tx=%lld grand_tx=%lld "
               "child_updates=%lld (%.0f commits/s)\n",
               StressSeconds, WarmupSeconds, NumThreads, CrossRatio,
               leaf_total_t, gcount_t, total_commits,
               total_commits / sec);
    } else {
        long long leaf_total_full = 0;
        for(int i = 0; i < NumThreads; ++i) leaf_total_full += leaf_count[i];
        long long total_full = gcount_full * (long long)NumThreads + leaf_total_full;
        printf("[3level_mixed MaxCommits=%d N=%d CrossRatio=%d] "
               "leaf_tx=%lld grand_tx=%lld elapsed=%.2fs "
               "child_updates=%lld (%.0f commits/s)\n",
               MaxCommits, NumThreads, CrossRatio,
               leaf_total_full, gcount_full, sec,
               total_full, total_full / sec);
    }

    if(!ok) return 1;
    fprintf(stderr, "PASS\n");
    return 0;
}
