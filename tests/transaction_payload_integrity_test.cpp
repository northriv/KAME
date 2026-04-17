/***************************************************************************
 * transaction_payload_integrity_test.cpp
 *
 * Mirrors TLA+ Layer 2 (BundleUnbundle_2level) TerminalPayloadCheck:
 * exhaustive-on-TLA+ / stress-on-hardware cross validation.
 *
 * Per-thread loop:
 *   1. CommitParent : parent-scope tr, +1 each child payload (triggers bundle)
 *   2. CommitChild  : direct tr per child, +1 payload (triggers unbundle)
 *
 * Terminal invariant (TerminalPayloadCheck):
 *   each child.m_x == (2 * commits_per_child) % MaxPayload
 *
 * Invoke:
 *   ./transaction_payload_integrity_test [StressSeconds] [NumThreads] [MaxPayload]
 *   StressSeconds=0 : correctness mode (MaxCommits=10000, check payload)
 *   StressSeconds>0 : benchmark mode (run N seconds, report commits/s)
 *
 * Benchmark comparison with test_bundle_2level.c:
 *   gcc -O2 -DSTRESS_SECONDS=10 -DNUM_THREADS=4 test_bundle_2level.c -lpthread
 *   ./transaction_payload_integrity_test 10 4
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

// Single node type for the whole tree (all same type, like transaction_test.cpp).
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
    int MaxCommits    = (StressSeconds > 0) ? 0x7fffffff : 10000;

    fprintf(stderr, "[payload_integrity] StressSeconds=%d NumThreads=%d MaxPayload=%d\n",
            StressSeconds, NumThreads, MaxPayload);

    // Tree construction: Parent -> {Child1, Child2}
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    shared_ptr<MyNode> child1(MyNode::create<MyNode>());
    shared_ptr<MyNode> child2(MyNode::create<MyNode>());
    parent->insert(child1);
    parent->insert(child2);

    std::atomic<int> barrier{0};
    std::atomic<bool> stop{false};
    std::atomic<long long> total_child1_commits{0};
    std::atomic<long long> total_child2_commits{0};

    auto worker = [&](int tid) {
        (void)tid;
        barrier.fetch_add(1);
        while (barrier.load() < NumThreads) std::this_thread::yield();

        long long my_c1 = 0, my_c2 = 0;
        unsigned int mp = (unsigned)MaxPayload;

        for (int iter = 0; iter < MaxCommits; ++iter) {
            if (StressSeconds > 0 && stop.load(std::memory_order_relaxed)) break;

            // --- CommitParent: parent-scope transaction touching both children ---
            parent->iterate_commit([&](Tr& tr) {
                tr[*child1].m_x = (tr[*child1].m_x + 1) % mp;
                tr[*child2].m_x = (tr[*child2].m_x + 1) % mp;
            });
            ++my_c1; ++my_c2;

            // --- CommitChild: direct per-child commits (unbundles if bundled) ---
            child1->iterate_commit([&](Tr& tr) {
                tr[*child1].m_x = (tr[*child1].m_x + 1) % mp;
            });
            ++my_c1;

            child2->iterate_commit([&](Tr& tr) {
                tr[*child2].m_x = (tr[*child2].m_x + 1) % mp;
            });
            ++my_c2;
        }

        total_child1_commits.fetch_add(my_c1);
        total_child2_commits.fetch_add(my_c2);
    };

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(NumThreads);
    for (int i = 0; i < NumThreads; ++i) threads.emplace_back(worker, i);

    if (StressSeconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(StressSeconds));
        stop.store(true, std::memory_order_release);
    }

    for (auto& t : threads) t.join();

    auto end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    long long cc1 = total_child1_commits.load();
    long long cc2 = total_child2_commits.load();

    Shot s1(*child1), s2(*child2);
    unsigned int got1 = s1[*child1].m_x;
    unsigned int got2 = s2[*child2].m_x;

    if (StressSeconds > 0) {
        // Stress mode: verify payload matches commit count, report throughput.
        unsigned int exp1 = (unsigned)(cc1 % MaxPayload);
        unsigned int exp2 = (unsigned)(cc2 % MaxPayload);
        printf("[payload_integrity stress %ds] Child1=%lld commits, Child2=%lld commits "
               "(total=%lld, %.0f commits/s)\n",
               StressSeconds, cc1, cc2, cc1 + cc2, (cc1 + cc2) / sec);
        if (got1 != exp1 || got2 != exp2) {
            fprintf(stderr, "FAIL: payload mismatch child1=%u(exp %u) child2=%u(exp %u)\n",
                    got1, exp1, got2, exp2);
            return 1;
        }
    } else {
        // Correctness mode: deterministic expected value.
        unsigned int expected = (2u * (unsigned)MaxCommits * (unsigned)NumThreads) % (unsigned)MaxPayload;
        printf("[payload_integrity MaxCommits=%d NumThreads=%d MaxPayload=%d]  "
               "child1=%u child2=%u expected=%u elapsed=%.2fs "
               "commits=%lld (%.0f commits/s)\n",
               MaxCommits, NumThreads, MaxPayload,
               got1, got2, expected, sec, cc1 + cc2, (cc1 + cc2) / sec);
        if (got1 != expected || got2 != expected) {
            fprintf(stderr, "FAIL: payload mismatch\n");
            return 1;
        }
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
