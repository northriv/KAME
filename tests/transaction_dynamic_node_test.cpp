/*
 * transaction_dynamic_node_test.cpp
 *
 *  Test code of software transactional memory, for simultaneous transaction
 *  including insertion/removal/swap of object links on tree-structure objects.
 *
 *  Topology (static backbone, dynamic extensions per worker):
 *
 *    gn1 -> gn2 -> gn3 -> leaf_in_gn3   (set up in main, static)
 *    gn4                                  (independent node, no static parent)
 *
 *  Each worker thread creates thread-local p1, p2 and on every 10th
 *  iteration builds a transient hard link:
 *
 *    p1 -> p2      (always, every iteration)
 *    gn2 -> p2     (10th: second parent for p2 — the hard link)
 *    gn1 -> p1     (10th: p2 reachable via gn1->p1->p2 AND gn1->gn2->p2)
 *
 *  Concurrent gn1->iterate_commit (spanning gn3 inside gn2's subtree, plus
 *  p2 on 10th) from multiple threads triggers the bundle self-collision:
 *  Phase 1's recursive collection of gn2's subtree walks p2's bundledBy
 *  chain back to gn1's own bundle wrapper → COLLIDED → null subpacket slot.
 *
 *  The test checks all payload sums remain zero (balanced increments /
 *  decrements) and no STM consistency assertion fires.
 */

#include "support_standalone.h"

#include <stdint.h>
#include <thread>

#include "transaction.h"

#include "xthread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;

//#define TRANSACTIONAL_STRICT_assert

class LongNode;
typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() : Transactional::Node<LongNode>() {
        ++objcnt;
    }
    virtual ~LongNode() {
        --objcnt;
    }

    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        Payload(const Payload &x)
            : Transactional::Node<LongNode>::Payload(x), m_x(x.m_x) {
            total += m_x;
        }
        virtual ~Payload() {
            total -= m_x;
        }
        operator long() const { return m_x; }
        Payload &operator=(const long &x) {
            total += x - m_x;
            m_x = x;
            return *this;
        }
        Payload &operator+=(const long &x) {
            total += x;
            m_x += x;
            return *this;
        }
    private:
        long m_x;
    };
};

template <class T>
typename std::enable_if<std::is_base_of<LongNode, T>::value,
    const typename Transactional::SingleSnapshot<LongNode, T> >::type
operator*(T &node) {
    return Transactional::SingleSnapshot<LongNode, T>(node);
}

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, gn2, gn3, gn4;

#define NUM_THREADS 2

static void
start_routine(void) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    shared_ptr<LongNode> p1(LongNode::create<LongNode>());
    shared_ptr<LongNode> p2(LongNode::create<LongNode>());
    for(int i = 0; i < 2500; i++) {
        p1->insert(p2);
        if((i % 10) == 0) {
            // Build hard link: p2 becomes child of gn2 while already a
            // child of p1.  With gn1->p1 also inserted, p2 is reachable
            // via two paths: gn1->p1->p2  AND  gn1->gn2->p2.
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
                if( !gn2->insert(tr1, p2))
                    return false;
                return true;
            });
            gn1->insert(p1);
        }
        // Forces gn1 to bundle its subtree spanning gn2->gn3 (and p2 on
        // 10th when hard link is active) — the self-collision window.
        // Net payload per iteration: gn3=+1, gn2=+1, p2=+1 on 10th;
        // gn1 net = 0 (write +1 then -1 against intermediate snapshot).
        gn1->iterate_commit([=](Transaction &tr1){
            Snapshot &ctr1(tr1);
            tr1[gn1] = ctr1[gn1] + 1;
            tr1[gn3] = ctr1[gn3] + 1;
            Snapshot &str1(tr1);
            tr1[gn1] = str1[gn1] - 1;
            tr1[gn2] = str1[gn2] + 1;
            if((i % 10) == 0)
                tr1[p2] = str1[p2] + 1;
        });
        // Independent gn4 transaction — scheduling pressure without
        // touching the gn1-gn3 bundle tree.
        gn4->iterate_commit([=](Transaction &tr1){
            tr1[gn4] = tr1[gn4] + 1;
            tr1[gn4] = tr1[gn4] - 1;
        });
        p1->release(p2);
        // Separate gn2-rooted transaction: concurrent bundling pressure on
        // gn2's subtree while it still holds p2 (not yet released from gn2).
        // Net payload per iteration: gn2=-1, gn3=-1, p2=-1 on 10th.
        gn2->iterate_commit([=](Transaction &tr1){
            Snapshot &str1(tr1);
            tr1[gn2] = tr1[gn2] - 1;
            tr1[gn3] = str1[gn3] - 1;
            if((i % 10) == 0)
                tr1[p2] = str1[p2] - 1;
        });
        if((i % 10) == 0) {
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
                if( !gn2->release(tr1, p2))
                    return false;
                return true;
            });
            gn1->release(p1);
        }
    }
}

int
main(int argc, char **argv) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int k = 0; k < 100; k++) {
        gn1.reset(LongNode::create<LongNode>());
        gn2.reset(LongNode::create<LongNode>());
        gn3.reset(LongNode::create<LongNode>());
        gn4.reset(LongNode::create<LongNode>());

        // Static backbone: gn1 -> gn2 -> gn3 -> leaf_in_gn3.
        // leaf_in_gn3 gives gn3 a non-trivial subtree depth so that gn2's
        // recursive bundle always has real work to do when collecting gn3.
        shared_ptr<LongNode> leaf_in_gn3(LongNode::create<LongNode>());
        gn3->insert(leaf_in_gn3);
        gn2->insert(gn3);
        gn1->insert(gn2);

        std::thread threads[NUM_THREADS];
        for(int i = 0; i < NUM_THREADS; i++) {
            std::thread th(&start_routine);
            threads[i].swap(th);
        }
        for(int i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
        printf("join\n");

        if(***gn1 || ***gn2 || ***gn3 || ***gn4) {
            printf("failed1\n");
            printf("Gn1:%ld\n", (long)***gn1);
            printf("Gn2:%ld\n", (long)***gn2);
            printf("Gn3:%ld\n", (long)***gn3);
            printf("Gn4:%ld\n", (long)***gn4);
            return -1;
        }

        // Reset in insertion-reverse order; ref-counting cascade destroys
        // the sub-tree when gn1's last holder is dropped.
        gn1.reset();
        gn2.reset();
        gn3.reset();
        gn4.reset();
        leaf_in_gn3.reset();

        if(objcnt != 0) {
            printf("failed1\n");
            return -1;
        }
        if(total != 0) {
            printf("failed total=%ld\n", (long)total);
            return -1;
        }
    }
    printf("succeeded\n");
    return 0;
}
