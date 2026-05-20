/*
 * transaction_hardlink_collision_test.cpp
 *
 * Minimal reproducer for the bundle self-collision bug.
 *
 * Topology (3 nodes, static — no dynamic insert/release):
 *
 *      gn1 (bundle root)
 *      ├── A          (depth-1 child)
 *      │   └── B      (depth-2 child)
 *      └── B          (depth-1 hard-link of B)
 *
 * Each thread repeatedly snapshots gn1 (via iterate_commit with a
 * payload touch on gn1), which triggers gn1 bundling.  When the bundle
 * recurses into A's Phase 1 to collect B, B's bundledBy chain leads
 * back to gn1 (because the prior bundle Phase-3'd B.m_link to
 * bundled_ref(gn1)).  Walking up hits gn1's wrapper carrying the
 * current bundle's serial → COLLIDED → subpacket_new=null → if
 * is_bundle_root forces missing=false, the published gn1 packet has a
 * null sub-slot whose subnode is not findable elsewhere in the tree.
 */

#include "support_standalone.h"

#include <stdint.h>
#include <thread>

#include "transaction.h"
#include "xthread.cpp"

class LongNode;
typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() : Transactional::Node<LongNode>() {}
    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        Payload(const Payload &x)
            : Transactional::Node<LongNode>::Payload(x), m_x(x.m_x) {}
        operator long() const { return m_x; }
        Payload &operator=(const long &x) { m_x = x; return *this; }
        Payload &operator+=(const long &x) { m_x += x; return *this; }
    private:
        long m_x;
    };
};

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, A, B;

#define NUM_THREADS 2
#define NUM_ITERATIONS 5000

static void
worker_root(void) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        // Multi-node transaction: forces gn1 bundling (must collect A
        // and B's packets to read/write B's payload).
        gn1->iterate_commit([](Transaction &tr) {
            tr[ *B] = (long)tr[ *B] + 1;
            tr[ *B] = (long)tr[ *B] - 1;
        });
    }
}

static void
worker_leaf(void) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        // Single-node transaction on B: forces unbundle of B from its
        // current parent (gn1 or A), which re-marks the upper tree
        // missing → the next gn1 snapshot re-bundles.  This is the
        // pressure that makes COLLIDED windows real.
        B->iterate_commit([](Transaction &tr) {
            tr[B] = (long)tr[B] + 1;
            tr[B] = (long)tr[B] - 1;
        });
    }
}

int
main(int argc, char **argv) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);

    for(int k = 0; k < 20; k++) {
        gn1.reset(LongNode::create<LongNode>());
        A.reset(LongNode::create<LongNode>());
        B.reset(LongNode::create<LongNode>());

        // Build topology: gn1 -> A -> B, and gn1 -> B (hard-link).
        gn1->insert(A);
        A->insert(B);
        gn1->insert(B);  // hardlink: B is now a direct child of gn1 too

        std::thread threads[NUM_THREADS];
        for(int t = 0; t < NUM_THREADS; t++) {
            // Alternate worker_root (bundle via gn1) and worker_leaf
            // (direct-commit B, forces unbundle/re-bundle cycle).
            std::thread th(t % 2 == 0 ? &worker_root : &worker_leaf);
            threads[t].swap(th);
        }
        for(int t = 0; t < NUM_THREADS; t++) {
            threads[t].join();
        }

        // Tear down (reverse order of insert)
        gn1->release(B);
        A->release(B);
        gn1->release(A);

        gn1.reset();
        A.reset();
        B.reset();
    }

    printf("succeeded\n");
    return 0;
}
