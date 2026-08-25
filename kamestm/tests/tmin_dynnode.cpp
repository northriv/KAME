/* transaction_dynamic_node_test, minimised.  NOT FINAL — see the open list.
 *
 * Verified removed (each A/B'd, thread count raised to hold the rate):
 *   - ComplexNode and the whole single-threaded setup block
 *   - every print_() / tr1.print() dump
 *   - the main thread's gn3->insert/release(gn4) churn
 *   - gn2->swap(p2, gn3)
 *   - gn4 (the detached node and its transaction)
 *   - the per-round tree teardown/rebuild: the tree is built ONCE.  What the
 *     outer loop is for is thread creation and exit; one round with the same
 *     total work does not fire.
 *
 * Still open (ablation running): the gn1 multi-node Tx, the gn2 Tx, the
 * p2-into-gn2 splice, the p1-into-gn1 splice, trans(*gn3), the private
 * p1/p2 churn.
 *
 * Needs, so far: a PERSISTENT tree, MANY concurrent threads, and REPEATED
 * thread create/exit under it.  Fires only with the pool allocator, only
 * when it was built by gcc at -O3 (one of seven -O3-only flags), and not at
 * all if freed addresses are kept out of circulation.
 *
 *   ./tmin 100 16 1250        # rounds threads iters -- ~24 s, fires ~2/3
 *
 * Ablation switches: A_NO_GN1TX A_NO_GN2TX A_NO_P2TREE A_NO_P1TREE
 *                    A_NO_TRANSGN3 A_NO_P1P2
 */
#include "support_standalone.h"
#include <stdint.h>
#include <thread>
#include <vector>
#include "transaction.h"
#include "xthread.cpp"

atomic<int> objcnt = 0;

class LongNode;
typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() : Transactional::Node<LongNode>() { ++objcnt; }
    virtual ~LongNode() { --objcnt; }
    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        Payload(const Payload &x) : Transactional::Node<LongNode>::Payload(x), m_x(x.m_x) {}
        operator long() const {return m_x;}
        Payload &operator=(const long &x) { m_x = x; return *this; }
        Payload &operator+=(const long &x) { m_x += x; return *this; }
    private:
        long m_x;
    };
};

#define trans(node) for(Transaction \
    implicit_tr(node, false); !implicit_tr.isModified() || !implicit_tr.commitOrNext(); ) implicit_tr[node]

template <class T>
typename std::enable_if<std::is_base_of<LongNode, T>::value,
    const typename Transactional::SingleSnapshot<LongNode, T> >::type
 operator*(T &node) { return Transactional::SingleSnapshot<LongNode, T>(node); }

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, gn2, gn3;
/* S_TWO_LEVEL: drop gn1 entirely; the splice transaction is rooted at gn2,
   so the shared tree is gn2<-gn3 (maps onto BundleUnbundle_2level_*). */
#ifdef S_TWO_LEVEL
  #define SPLICE_ROOT gn2
  #define A_NO_GN1TX 1
  #define A_NO_P1TREE 1
#else
  #define SPLICE_ROOT gn1
#endif
static int g_iters = 1250;

void start_routine(void) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    shared_ptr<LongNode> p1(LongNode::create<LongNode>());
    shared_ptr<LongNode> p2(LongNode::create<LongNode>());
    for(int i = 0; i < g_iters; i++) {
#ifndef A_NO_P1P2
        p1->insert(p2);
#endif
        if((i % 10) == 0) {
#ifndef A_NO_P2TREE
            SPLICE_ROOT->iterate_commit_if([=](Transaction &tr1)->bool{
                return gn2->insert(tr1, p2);
            });
#endif
#ifndef A_NO_P1TREE
            gn1->insert(p1);
#endif
        }
#ifndef A_NO_GN1TX
        gn1->iterate_commit([=](Transaction &tr1){
            Snapshot &ctr1(tr1);
            tr1[gn1] = ctr1[gn1] + 1;
            tr1[gn3] = ctr1[gn3] + 1;
            Snapshot &str1(tr1);
            tr1[gn1] = str1[gn1] - 1;
            tr1[gn2] = str1[gn2] + 1;
        });
#endif
#ifndef A_NO_TRANSGN3
        trans(*gn3) += 1;
#endif
#ifndef A_NO_P1P2
        p1->release(p2);
#endif
#ifndef A_NO_GN2TX
        gn2->iterate_commit([=](Transaction &tr1){
            Snapshot &str1(tr1);
            tr1[gn2] = tr1[gn2] - 1;
            tr1[gn3] = str1[gn3] - 1;
        });
#endif
#ifndef A_NO_TRANSGN3
        trans(*gn3) += -1;
#endif
        if((i % 10) == 0) {
#ifndef A_NO_P2TREE
            SPLICE_ROOT->iterate_commit_if([=](Transaction &tr1)->bool{
                return gn2->release(tr1, p2);
            });
#endif
#ifndef A_NO_P1TREE
            gn1->release(p1);
#endif
        }
    }
}

int main(int argc, char **argv) {
    int rounds  = (argc > 1) ? atoi(argv[1]) : 100;
    int nthread = (argc > 2) ? atoi(argv[2]) : 16;
    g_iters     = (argc > 3) ? atoi(argv[3]) : 1250;
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);

#ifndef S_TWO_LEVEL
    gn1.reset(LongNode::create<LongNode>());
#endif
    gn2.reset(LongNode::create<LongNode>());
    gn3.reset(LongNode::create<LongNode>());
#ifndef S_TWO_LEVEL
    gn1->insert(gn2);
#endif
    gn2->insert(gn3);

    for(int k = 0; k < rounds; k++) {
        std::vector<std::thread> threads;
        for(int i = 0; i < nthread; i++) threads.emplace_back( &start_routine);
        for(auto &t : threads) t.join();
    }

    gn1.reset(); gn2.reset(); gn3.reset();
    if(objcnt != 0) { printf("failed objcnt=%d\n", (int)objcnt); return -1; }
    printf("succeeded\n");
    return 0;
}
