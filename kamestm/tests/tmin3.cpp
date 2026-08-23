/* transaction_dynamic_node_test, minimised. */
#include "support_standalone.h"
#include <stdint.h>
#include <thread>
#include <vector>
#include "transaction.h"
#include "xthread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;

class LongNode;
typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() : Transactional::Node<LongNode>() { ++objcnt; }
    virtual ~LongNode() { --objcnt; }
    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        Payload(const Payload &x) : Transactional::Node<LongNode>::Payload(x), m_x(x.m_x) { total += m_x; }
        virtual ~Payload() { total -= m_x; }
        operator long() const {return m_x;}
        Payload &operator=(const long &x) { total += x - m_x; m_x = x; return *this; }
        Payload &operator+=(const long &x) { total += x; m_x += x; return *this; }
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

shared_ptr<LongNode> gn1, gn2, gn3, gn4;
static int g_iters = 2500;

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
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
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
#ifndef A_NO_P2TREE
            if((i % 10) == 0) tr1[p2] = str1[p2] + 1;
#endif
        });
#endif
#ifndef A_NO_TRANSGN3
        trans(*gn3) += 1;
#endif
#ifndef A_NO_GN4
        gn4->iterate_commit([=](Transaction &tr1){
            tr1[gn4] = tr1[gn4] + 1;
            tr1[gn4] = tr1[gn4] - 1;
        });
#endif
#ifndef A_NO_P1P2
        p1->release(p2);
#endif
#ifndef A_NO_GN2TX
        gn2->iterate_commit([=](Transaction &tr1){
            Snapshot &str1(tr1);
            tr1[gn2] = tr1[gn2] - 1;
            tr1[gn3] = str1[gn3] - 1;
#ifndef A_NO_P2TREE
            if((i % 10) == 0) tr1[p2] = str1[p2] - 1;
#endif
        });
#endif
#ifndef A_NO_TRANSGN3
        trans(*gn3) += -1;
#endif
        if((i % 10) == 0) {
#ifndef A_NO_P2TREE
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
                return gn2->release(tr1, p2);
            });
#endif
#ifndef A_NO_P1TREE
            gn1->release(p1);
#endif
        }
    }
    return;
}

int main(int argc, char **argv) {
    int rounds  = (argc > 1) ? atoi(argv[1]) : 100;
    int nthread = (argc > 2) ? atoi(argv[2]) : 8;
    g_iters     = (argc > 3) ? atoi(argv[3]) : 2500;
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int k = 0; k < rounds; k++) {
        gn1.reset(LongNode::create<LongNode>());
        gn2.reset(LongNode::create<LongNode>());
        gn3.reset(LongNode::create<LongNode>());
        gn4.reset(LongNode::create<LongNode>());
        gn1->insert(gn2);
        gn2->insert(gn3);

        std::vector<std::thread> threads;
        for(int i = 0; i < nthread; i++) threads.emplace_back( &start_routine);
        for(auto &t : threads) t.join();

#ifndef NO_VALUE_CHECKS
        if(***gn1 || ***gn2 || ***gn3 || ***gn4) {
            printf("failed values Gn1:%ld Gn2:%ld Gn3:%ld Gn4:%ld\n",
                (long)***gn1, (long)***gn2, (long)***gn3, (long)***gn4);
            return -1;
        }
#endif
        gn1.reset(); gn2.reset(); gn3.reset(); gn4.reset();
        if(objcnt != 0) { printf("failed objcnt=%d\n", (int)objcnt); return -1; }
#ifndef NO_VALUE_CHECKS
        if(total != 0) { printf("failed total=%ld\n", (long)total); return -1; }
#endif
    }
    printf("succeeded\n");
    return 0;
}
