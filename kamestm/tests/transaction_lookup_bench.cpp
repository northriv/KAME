/*
 * transaction_lookup_bench.cpp
 *
 * Microbenchmark for Snapshot::at() / Transaction::operator[] lookup cost,
 * sized to expose the 1-entry lookup memo (Snapshot<XN>::LookupMemo):
 *
 *   snap/same      repeated shot[*child] on one node      (memo best case)
 *   snap/alt2      alternating two children               (memo worst case)
 *   snap/root      repeated shot[*root]                   (pre-existing O(1)
 *                                                          fast path; must
 *                                                          not regress)
 *   snap/stale     repeated shot[*child] on a snapshot taken BEFORE many
 *                  commits moved the live tree on (hint walk fails; without
 *                  the memo every access pays an O(tree) forwardLookup)
 *   tr/same        repeated tr[*child] within one transaction body
 *   tr/alt2        alternating two children within one transaction body
 *
 * Prints ns/op per scenario; always exits 0 (perf tool, not a testcase).
 */

#include "support_standalone.h"

#include <stdint.h>
#include <chrono>
#include <thread>

#include "transaction.h"
#include "transaction_impl.h"

#include "xthread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() { ++objcnt; }
    virtual ~LongNode() { --objcnt; }

    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        long m_x;
    };
};

using Snapshot = Transactional::Snapshot<LongNode>;
using Transaction = Transactional::Transaction<LongNode>;

static long s_sink = 0;

template <typename Body>
static double bench_ns(const char *label, long iters, Body &&body) {
    using clk = std::chrono::steady_clock;
    body(iters / 100 + 1);   // warmup
    auto t0 = clk::now();
    body(iters);
    auto t1 = clk::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    printf("%-12s %8.2f ns/op\n", label, ns);
    return ns;
}

int main() {
    constexpr int NUM_CHILDREN = 16;
    constexpr long N_READ = 20000000;
    constexpr long N_WRITE = 5000000;

    shared_ptr<LongNode> root(LongNode::create<LongNode>());
    shared_ptr<LongNode> children[NUM_CHILDREN];
    for(int i = 0; i < NUM_CHILDREN; ++i) {
        children[i] = shared_ptr<LongNode>(LongNode::create<LongNode>());
        root->insert(children[i]);
    }
    shared_ptr<LongNode> a = children[3], b = children[12];

    root->iterate_commit([&](Transaction &tr) {
        for(int i = 0; i < NUM_CHILDREN; ++i)
            tr[ *children[i]].m_x = i;
    });

    {
        Snapshot shot( *root);
        bench_ns("snap/same", N_READ, [&](long n) {
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *a].m_x;
        });
        bench_ns("snap/alt2", N_READ, [&](long n) {
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *(i & 1 ? a : b)].m_x;
        });
        bench_ns("snap/alt4", N_READ, [&](long n) {
            // 4-node rotation == LookupMemo::SLOTS: should stay memoized.
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *children[(i & 3) * 4]].m_x;
        });
        bench_ns("snap/rr8", N_READ / 4, [&](long n) {
            // 8-node rotation > SLOTS: every access evicts — true-miss cost.
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *children[(i & 7) * 2]].m_x;
        });
        bench_ns("snap/root", N_READ, [&](long n) {
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *root].m_x;
        });
    }

    {
        // Stale snapshot: take it, then move the live tree on with many
        // independent commits so the per-access bundledBy hint walk no
        // longer matches the frozen packets.
        Snapshot shot( *root);
        for(int i = 0; i < 1000; ++i) {
            a->iterate_commit([&](Transaction &tr) {
                tr[ *a].m_x += 1;
            });
        }
        bench_ns("snap/stale", N_READ / 10, [&](long n) {
            for(long i = 0; i < n; ++i)
                s_sink += shot[ *a].m_x;
        });
    }

    {
        Transaction tr( *root);
        bench_ns("tr/same", N_WRITE, [&](long n) {
            for(long i = 0; i < n; ++i)
                tr[ *a].m_x += 1;
        });
        bench_ns("tr/alt2", N_WRITE, [&](long n) {
            for(long i = 0; i < n; ++i)
                tr[ *(i & 1 ? a : b)].m_x += 1;
        });
        bench_ns("tr/alt4", N_WRITE, [&](long n) {
            for(long i = 0; i < n; ++i)
                tr[ *children[(i & 3) * 4]].m_x += 1;
        });
        // transaction intentionally dropped without commit
    }

    fprintf(stderr, "done (sink=%ld)\n", s_sink);
    return 0;
}
