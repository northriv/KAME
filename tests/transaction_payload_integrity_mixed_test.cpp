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
    std::atomic<bool> stop{false};
    std::vector<long long> leaf_count(NumThreads, 0);
    std::atomic<long long> parent_count_total{0};

    auto worker = [&](int tid) {
        barrier.fetch_add(1);
        while(barrier.load() < NumThreads) std::this_thread::yield();

        long long my_leaf = 0;
        long long my_parent = 0;
        unsigned int mp = (unsigned)MaxPayload;
        shared_ptr<MyNode> &my_leaf_node = children[tid];

        for(int iter = 0; iter < MaxCommits; ++iter) {
            if(StressSeconds > 0 && stop.load(std::memory_order_relaxed)) break;

            bool do_parent = (CrossRatio > 0) && ((iter % CrossRatio) == 0);
            if(do_parent) {
                // Parent-scope tx — bundles all children; every child's
                // m_x increments by 1. This is the cross-thread contention
                // point; N concurrent Parent commits serialize at Parent's
                // Linkage.
                parent->iterate_commit([&](Tr& tr) {
                    for(int c = 0; c < NumThreads; ++c) {
                        tr[*children[c]].m_x = (tr[*children[c]].m_x + 1) % mp;
                    }
                });
                ++my_parent;
            } else {
                // Disjoint leaf commit — only this thread's own child.
                // No cross-thread contention (other threads touch distinct
                // children). 1 CAS per commit on the leaf Linkage.
                my_leaf_node->iterate_commit([&](Tr& tr) {
                    tr[*my_leaf_node].m_x = (tr[*my_leaf_node].m_x + 1) % mp;
                });
                ++my_leaf;
            }
        }

        leaf_count[tid] = my_leaf;
        parent_count_total.fetch_add(my_parent);
    };

    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(NumThreads);
    for(int i = 0; i < NumThreads; ++i) threads.emplace_back(worker, i);

    if(StressSeconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(StressSeconds));
        stop.store(true, std::memory_order_release);
    }

    for(auto& t : threads) t.join();

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();

    long long pcount = parent_count_total.load();
    // Throughput convention matches the existing transaction_payload_integrity_test:
    // count child-touches, not tx invocations. Each Parent-scope tx touches N
    // children (one increment per child inside the bundle), so it contributes N
    // to total_commits. This makes K={0, 10, 1} numbers directly comparable:
    // rate_K=1 / rate_K=0 is the "strong-vs-disjoint" ratio per child update,
    // not an artefact of the counting granularity.
    long long total_commits = pcount * (long long)NumThreads;
    for(int i = 0; i < NumThreads; ++i) total_commits += leaf_count[i];

    // Verify terminal payload: each child[i].m_x == (leaf[i] + pcount) % mp.
    bool ok = true;
    for(int i = 0; i < NumThreads; ++i) {
        Shot s(*children[i]);
        unsigned int got = s[*children[i]].m_x;
        unsigned int expected = (unsigned)((leaf_count[i] + pcount) % MaxPayload);
        if(got != expected) {
            fprintf(stderr, "FAIL: child[%d].m_x=%u expected=%u "
                           "(leaf=%lld parent_total=%lld)\n",
                    i, got, expected, leaf_count[i], pcount);
            ok = false;
        }
    }

    // Report leaf/parent tx counts (raw invocations) alongside the
    // child-touch-based total rate.
    long long leaf_total = 0;
    for(int i = 0; i < NumThreads; ++i) leaf_total += leaf_count[i];
    if(StressSeconds > 0) {
        printf("[payload_integrity_mixed stress %ds] N=%d CrossRatio=%d "
               "leaf_tx=%lld parent_tx=%lld "
               "child_updates=%lld (%.0f commits/s)\n",
               StressSeconds, NumThreads, CrossRatio,
               leaf_total, pcount, total_commits,
               total_commits / sec);
    } else {
        printf("[payload_integrity_mixed MaxCommits=%d N=%d CrossRatio=%d] "
               "leaf_tx=%lld parent_tx=%lld elapsed=%.2fs "
               "child_updates=%lld (%.0f commits/s)\n",
               MaxCommits, NumThreads, CrossRatio,
               leaf_total, pcount, sec,
               total_commits, total_commits / sec);
    }

    if(!ok) return 1;
    fprintf(stderr, "PASS\n");
    return 0;
}
