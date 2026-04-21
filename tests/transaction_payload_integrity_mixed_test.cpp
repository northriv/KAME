/***************************************************************************
 * transaction_payload_integrity_mixed_test.cpp
 *
 * Synchrobench-style variable-contention benchmark: each thread owns its
 * own leaf child (Child_0..Child_{N-1} under a single Parent), and the
 * `CrossRatio` knob controls how often a thread issues a Parent-scope
 * transaction (which touches all children and creates cross-thread
 * contention) versus a leaf-only transaction (disjoint, no contention).
 *
 * Regime coverage:
 *   CrossRatio = 0 : pure disjoint — every tx is a leaf commit on the
 *                    thread's own child. Upper bound on KAME throughput;
 *                    bundle/unbundle never fires; per-commit is 1 CAS.
 *   CrossRatio = 1 : pure Parent-scope — every tx bundles all children.
 *                    Matches the existing transaction_payload_integrity_test
 *                    worst-case when N=NumThreads.
 *   CrossRatio = K : 1-in-K iterations is Parent-scope; the remaining
 *                    K-1 are disjoint leaf commits. Mirrors Synchrobench's
 *                    "update rate" knob at discrete ratios (K=10 ≈ 10 %
 *                    Parent, K=100 ≈ 1 % Parent).
 *
 * Terminal invariant after the run (both stress and correctness modes):
 *   child[i].m_x == (leaf_count_for_thread_i + parent_count_total) % MaxPayload
 *   where parent_count_total = sum of parent commits across ALL threads,
 *   since every Parent-scope commit increments EVERY child's m_x.
 *
 * Invoke:
 *   ./transaction_payload_integrity_mixed_test [StressSeconds] [NumThreads]
 *                                              [MaxPayload] [CrossRatio]
 *   StressSeconds=0 : correctness mode (MaxCommits=10000, check payload)
 *   StressSeconds>0 : benchmark mode (run N seconds, report commits/s)
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
    int CrossRatio    = (argc > 4) ? std::atoi(argv[4]) : 0;   // 0 = pure disjoint
    int MaxCommits    = (StressSeconds > 0) ? 0x7fffffff : 10000;

    if(NumThreads < 1) NumThreads = 1;
    if(MaxPayload < 1) MaxPayload = 1;
    if(CrossRatio < 0) CrossRatio = 0;

    fprintf(stderr, "[payload_integrity_mixed] StressSeconds=%d NumThreads=%d "
                    "MaxPayload=%d CrossRatio=%d\n",
            StressSeconds, NumThreads, MaxPayload, CrossRatio);

    // Tree: Parent -> N children (one leaf per thread).
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    std::vector<shared_ptr<MyNode>> children(NumThreads);
    for(int i = 0; i < NumThreads; ++i) {
        children[i].reset(MyNode::create<MyNode>());
        parent->insert(children[i]);
    }

    std::atomic<int> barrier{0};
    std::atomic<bool> warming_up{true};
    std::atomic<bool> stop{false};
    std::vector<long long> leaf_count(NumThreads, 0);
    std::vector<long long> leaf_count_timed(NumThreads, 0);
    std::atomic<long long> parent_count_total{0};
    std::atomic<long long> parent_count_timed{0};

    auto worker = [&](int tid) {
        barrier.fetch_add(1);
        while(barrier.load() < NumThreads) std::this_thread::yield();

        long long my_leaf = 0,   my_leaf_t = 0;
        long long my_parent = 0, my_parent_t = 0;
        unsigned int mp = (unsigned)MaxPayload;
        shared_ptr<MyNode> &my_leaf_node = children[tid];

        for(int iter = 0; iter < MaxCommits; ++iter) {
            if(StressSeconds > 0 && stop.load(std::memory_order_relaxed)) break;
            bool warming = warming_up.load(std::memory_order_relaxed);

            bool do_parent = (CrossRatio > 0) && ((iter % CrossRatio) == 0);
            if(do_parent) {
                parent->iterate_commit([&](Tr& tr) {
                    for(int c = 0; c < NumThreads; ++c) {
                        tr[*children[c]].m_x = (tr[*children[c]].m_x + 1) % mp;
                    }
                });
                ++my_parent;
                if(!warming) ++my_parent_t;
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
        parent_count_total.fetch_add(my_parent);
        parent_count_timed.fetch_add(my_parent_t);
    };

    std::vector<std::thread> threads;
    threads.reserve(NumThreads);
    for(int i = 0; i < NumThreads; ++i) threads.emplace_back(worker, i);

    // Warmup (1 s in stress mode) — workers run the workload under
    // warming_up=true so their timed counters stay at zero. Lets any
    // bundle/unbundle settlement finish before the timed phase starts.
    const int WarmupSeconds = (StressSeconds > 0) ? 1 : 0;
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

    long long pcount_full = parent_count_total.load();
    // Terminal payload-equivalence uses life-of-test counts (warmup
    // commits incremented the payload too).
    bool ok = true;
    for(int i = 0; i < NumThreads; ++i) {
        Shot s(*children[i]);
        unsigned int got = s[*children[i]].m_x;
        unsigned int expected = (unsigned)((leaf_count[i] + pcount_full) % MaxPayload);
        if(got != expected) {
            fprintf(stderr, "FAIL: child[%d].m_x=%u expected=%u "
                           "(leaf=%lld parent_total=%lld)\n",
                    i, got, expected, leaf_count[i], pcount_full);
            ok = false;
        }
    }

    // Reported rates use TIMED counters only (warmup excluded).
    long long pcount_t = parent_count_timed.load();
    long long leaf_total_t = 0;
    for(int i = 0; i < NumThreads; ++i) leaf_total_t += leaf_count_timed[i];
    long long total_commits = pcount_t * (long long)NumThreads + leaf_total_t;
    if(StressSeconds > 0) {
        printf("[payload_integrity_mixed stress %ds warmup %ds] N=%d CrossRatio=%d "
               "leaf_tx=%lld parent_tx=%lld "
               "child_updates=%lld (%.0f commits/s)\n",
               StressSeconds, WarmupSeconds, NumThreads, CrossRatio,
               leaf_total_t, pcount_t, total_commits,
               total_commits / sec);
    } else {
        long long leaf_total_full = 0;
        for(int i = 0; i < NumThreads; ++i) leaf_total_full += leaf_count[i];
        long long total_full = pcount_full * (long long)NumThreads + leaf_total_full;
        printf("[payload_integrity_mixed MaxCommits=%d N=%d CrossRatio=%d] "
               "leaf_tx=%lld parent_tx=%lld elapsed=%.2fs "
               "child_updates=%lld (%.0f commits/s)\n",
               MaxCommits, NumThreads, CrossRatio,
               leaf_total_full, pcount_full, sec,
               total_full, total_full / sec);
    }

    if(!ok) return 1;
    fprintf(stderr, "PASS\n");
    return 0;
}
