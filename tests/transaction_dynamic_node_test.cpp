/*
 * transaction_dynamic_node_test.cpp
 *
 *  Test code of software transactional memory, for simultaneous transaction including
 *  insertion/removal/swap of object links on tree-structure objects.
 */

#include "support.h"

#include <stdint.h>
#include <thread>

#include "transaction.h"

#include "xthread.cpp"

atomic<int> objcnt = 0; //# of living objects.
atomic<long> total = 0; //The sum of payloads.

//#define TRANSACTIONAL_STRICT_assert

class LongNode;
typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
	LongNode() : Transactional::Node<LongNode>() {
		++objcnt;
	//	trans(*this) = 0;
	}
	virtual ~LongNode() {
		--objcnt;
	}

	//! Data holder.
	struct Payload : public Transactional::Node<LongNode>::Payload {
		Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
		Payload(const Payload &x) : Transactional::Node<LongNode>::Payload(x), m_x(x.m_x) {
			total += m_x;
		}
		virtual ~Payload() {
			total -= m_x;
		}
		operator long() const {return m_x;}
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
//		double load[10000];
	};
};

class ComplexNode : public LongNode {
public:
	ComplexNode(Transaction &tr, shared_ptr<LongNode> &var) : LongNode(), m_var(var) {
//		msecsleep(40);
        m_1.reset(create<LongNode>());
		insert(m_1);
		var->insert(tr, m_1, false);
//		msecsleep(40);
//		m_2.reset(create<LongNode>());
//		insert(m_2);
//		m_3.reset(create<LongNode>());
//		insert(m_3);
//		m_4.reset(create<LongNode>());
//		msecsleep(40);
//		var->insert(tr, m_4);
	}
	virtual ~ComplexNode() {
		m_var->release(m_1);
//		m_var->release(m_4);
	}
	const shared_ptr<LongNode> &n1() const {return m_1;}
private:
	const shared_ptr<LongNode> m_var;
	shared_ptr<LongNode> m_1, m_2, m_3, m_4;
};

#define trans(node) for(Transaction \
	implicit_tr(node, false); !implicit_tr.isModified() || !implicit_tr.commitOrNext(); ) implicit_tr[node]

template <class T>
typename std::enable_if<std::is_base_of<LongNode, T>::value,
	const typename Transactional::SingleSnapshot<LongNode, T> >::type
 operator*(T &node) {
	return Transactional::SingleSnapshot<LongNode, T>(node);
}

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, gn2, gn3, gn4;

void
start_routine(void) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    printf("start\n");
	shared_ptr<LongNode> p1(LongNode::create<LongNode>());
	shared_ptr<LongNode> p2(LongNode::create<LongNode>());
    for(int i = 0; i < 2500; i++) {
		p1->insert(p2);
		if((i % 10) == 0) {
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
				if( !gn2->insert(tr1, p2))
                    return false;
                return true;
            });
			gn2->swap(p2, gn3);
			gn1->insert(p1);
		}
		//		gn1->print_();
        gn1->iterate_commit([=](Transaction &tr1){
//			tr1.print();
			Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			tr1[gn3] = ctr1[gn3] + 1;
            Snapshot &str1(tr1);
			tr1[gn1] = str1[gn1] - 1;
			tr1[gn2] = str1[gn2] + 1;
			if((i % 10) == 0) {
				tr1[p2] = str1[p2] + 1;
			}
//			printf("f");
        });
		trans(*gn3) += 1;
        gn4->iterate_commit([=](Transaction &tr1){
			tr1[gn4] = tr1[gn4] + 1;
			tr1[gn4] = tr1[gn4] - 1;
//			printf("f");
        });
		p1->release(p2);
        gn2->iterate_commit([=](Transaction &tr1){
            Snapshot &str1(tr1);
			tr1[gn2] = tr1[gn2] - 1;
			tr1[gn3] = str1[gn3] - 1;
			if((i % 10) == 0) {
				tr1[p2] = str1[p2] - 1;
			}
//			printf("f");
        });
		trans(*gn3) += -1;
		if((i % 10) == 0) {
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
				if( !gn2->release(tr1, p2))
                    return false;
                return true;
            });
			gn1->release(p1);
		}
	}
	long y = ***p2;
	if(y != 0) {
		printf("Error! P2=%ld\n", y);
		abort();
	}
	else
		printf("finish\n");
    return;
}

#define NUM_THREADS 4

int
main(int argc, char **argv) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int k = 0; k < 100; k++) {
        gn1.reset(LongNode::create<LongNode>());
        gn2.reset(LongNode::create<LongNode>());
        gn3.reset(LongNode::create<LongNode>());
        gn4.reset(LongNode::create<LongNode>());

        gn1->iterate_commit_if([=](Transaction &tr1)->bool{
            printf("1");
			if( !gn1->insert(tr1, gn2, true))
                return false;
			tr1[ *gn2] = tr1[ *gn2] + 1;
			if( !gn2->insert(tr1, gn3, true))
                return false;
			tr1.print();
			if( !gn3->insert(tr1, gn4, true))
                return false;
			tr1.print();
			if( !gn3->release(tr1, gn4))
                return false;
			tr1.print();
            return true;
        });
		gn1->print_();
		gn1->release(gn2);
		gn1->print_();
		gn1->insert(gn2);
		gn1->print_();
        gn2->iterate_commit([=](Transaction &tr1){
            printf("2");
			tr1[ *gn2] = tr1[ *gn2] - 1;
			tr1[ *gn3] = 0;
        });
		{
			Snapshot shot1(*gn1);
			shot1.print();
			long x = shot1[*gn3];
			printf("Gn3:%ld\n", x);
		}
		trans(*gn3) = 3;
		long x = ***gn3;
		printf("Gn3:%ld\n", x);
		trans(*gn3) = 0;

		shared_ptr<LongNode> p1(LongNode::create<LongNode>());
		gn3->insert(p1);
		gn1->insert(p1);
		gn1->swap(p1, gn2);
		trans(*gn1) = 3;
		trans(*gn1) = 0;

		{
			shared_ptr<LongNode> p2(LongNode::create<LongNode>());
			shared_ptr<LongNode> p22(LongNode::create<LongNode>());
			shared_ptr<LongNode> p211(LongNode::create<LongNode>());
			shared_ptr<LongNode> p21(LongNode::create<LongNode>());
			p2->insert(p21);
			p21->insert(p211);
			p2->insert(p211);
			trans(*p211) = 1;
			p21->insert(p22);
			p211->insert(p22);

			gn1->insert(p2);
			gn3->insert(p2);
			gn1->release(p2);
			gn3->release(p2);

			shared_ptr<ComplexNode> p2111;

            gn3->iterate_commit_if([=, &p2111](Transaction &tr1)->bool{
				printf("3");
				if( !p1->insert(tr1, p22, true))
                    return false;
				if( !gn3->insert(tr1, p2, true))
                    return false;

                p2111.reset(LongNode::create<ComplexNode>(ref(tr1), gn3));
				shared_ptr<LongNode> p2112(LongNode::create<LongNode>());
				shared_ptr<LongNode> p2113(LongNode::create<LongNode>());
				shared_ptr<LongNode> p2114(LongNode::create<LongNode>());
				p2111->insert(p2112);
				p2112->insert(p2113);
				p2111->insert(p2114);
				trans( *p2113) = 1;
				trans( *p2114) = 1;

				if( !p21->insert(tr1, p2111, true))
                    return false;
				if( !gn3->insert(tr1, p2111, true))
                    return false;

				tr1[ *p2113] = 0;
				tr1[ *p2114] = 0;

				tr1[*p22] = 1;
                return true;
            });
			{
				Snapshot shot1(*gn3);
				shot1[ *p2111->n1()];
			}
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
                printf("4");
				if( !gn3->insert(tr1, gn4, true))
                    return false;
				if( !gn2->insert(tr1, gn4, true))
                    return false;
                return true;
            });
            gn1->iterate_commit_if([=](Transaction &tr1)->bool{
				printf("5");
				if( !gn3->release(tr1, gn4))
                    return false;
				if( !gn2->release(tr1, gn4))
                    return false;
                return true;
            });
			{
				Snapshot shot1(*gn3);
				shot1[ *p2];
				shot1[ *p21];
				shot1[ *p22];
			}
			trans(*p211) = 0;
            gn3->iterate_commit_if([=](Transaction &tr1)->bool{
                if( !p1->release(tr1, p22))
                    return false;
                if( !gn3->release(tr1, p2))
                    return false;
                if( !gn3->release(tr1, p2111))
                    return false;
                return true;
            });
			trans(*p22) = 0;

            gn1->iterate_commit([=](Transaction &tr1){
                Snapshot &ctr1(tr1); // For reading.
				tr1[gn1] = ctr1[gn1] + 1;
				tr1[gn3] = ctr1[gn3] + 1;
                Snapshot &str1(tr1);
				tr1[gn1] = str1[gn1] - 1;
				tr1[gn3] = str1[gn3] - 1;
            });
		}
		gn1->print_();
		gn1->release(p1);
		gn1->print_();

        std::thread threads[NUM_THREADS];

        for(int i = 0; i < NUM_THREADS; i++) {
            std::thread th( &start_routine);
            threads[i].swap(th);
        }
        msecsleep(10);
        for(int i = 0; i < 100; i++) {
			gn3->insert(gn4);
			gn3->release(gn4);
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

		gn1.reset();
		gn2.reset();
		gn3.reset();
		gn4.reset();
		p1.reset();

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
