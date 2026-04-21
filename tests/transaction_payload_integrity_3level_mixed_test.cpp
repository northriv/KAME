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

    fprintf(stderr, "[3level_mixed] StressSeconds=%d NumThreads=%d "
                    "MaxPayload=%d CrossRatio=%d\n",
            StressSeconds, NumThreads, MaxPayload, CrossRatio);

    // Tree: Grand -> Parent -> N children.
    shared_ptr<MyNode> grand(MyNode::create<MyNode>());
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    grand->insert(parent);
    std::vector<shared_ptr<MyNode>> children(NumThreads);
    for(int i = 0; i < NumThreads; ++i) {
        children[i].reset(MyNode::create<MyNode>());
        parent->insert(children[i]);
    }

    std::atomic<int> barrier{0};
    std::atomic<bool> stop{false};
    std::vector<long long> leaf_count(NumThreads, 0);
    std::atomic<long long> grand_count_total{0};

    auto worker = [&](int tid) {
        barrier.fetch_add(1);
        while(barrier.load() < NumThreads) std::this_thread::yield();

        long long my_leaf = 0;
        long long my_grand = 0;
        unsigned int mp = (unsigned)MaxPayload;
        shared_ptr<MyNode> &my_leaf_node = children[tid];

        for(int iter = 0; iter < MaxCommits; ++iter) {
            if(StressSeconds > 0 && stop.load(std::memory_order_relaxed)) break;

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
            } else {
                my_leaf_node->iterate_commit([&](Tr& tr) {
                    tr[*my_leaf_node].m_x = (tr[*my_leaf_node].m_x + 1) % mp;
                });
                ++my_leaf;
            }
        }

        leaf_count[tid] = my_leaf;
        grand_count_total.fetch_add(my_grand);
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

    long long gcount = grand_count_total.load();
    // Throughput convention matches transaction_payload_integrity_test:
    // count child-touches, not tx invocations. Each Grand-scope tx touches
    // N children inside the bundle, so it contributes N to total_commits.
    // Makes rates directly comparable across K values per child update.
    long long total_commits = gcount * (long long)NumThreads;
    for(int i = 0; i < NumThreads; ++i) total_commits += leaf_count[i];

    bool ok = true;
    for(int i = 0; i < NumThreads; ++i) {
        Shot s(*children[i]);
        unsigned int got = s[*children[i]].m_x;
        unsigned int expected = (unsigned)((leaf_count[i] + gcount) % MaxPayload);
        if(got != expected) {
            fprintf(stderr, "FAIL: child[%d].m_x=%u expected=%u "
                           "(leaf=%lld grand_total=%lld)\n",
                    i, got, expected, leaf_count[i], gcount);
            ok = false;
        }
    }

    long long leaf_total = 0;
    for(int i = 0; i < NumThreads; ++i) leaf_total += leaf_count[i];
    if(StressSeconds > 0) {
        printf("[3level_mixed stress %ds] N=%d CrossRatio=%d "
               "leaf_tx=%lld grand_tx=%lld "
               "child_updates=%lld (%.0f commits/s)\n",
               StressSeconds, NumThreads, CrossRatio,
               leaf_total, gcount, total_commits,
               total_commits / sec);
    } else {
        printf("[3level_mixed MaxCommits=%d N=%d CrossRatio=%d] "
               "leaf_tx=%lld grand_tx=%lld elapsed=%.2fs "
               "child_updates=%lld (%.0f commits/s)\n",
               MaxCommits, NumThreads, CrossRatio,
               leaf_total, gcount, sec,
               total_commits, total_commits / sec);
    }

    if(!ok) return 1;
    fprintf(stderr, "PASS\n");
    return 0;
}
