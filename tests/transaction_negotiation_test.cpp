/*
 * transaction_negotiation_test.cpp
 *
 * Test code of software transactional memory, for simultaneous transaction with different periodicities.
 * The slower transaction would never succeed unless a proper wait is put on the faster loop.
 */

#include "support.h"


//For inline expansion of lock-free custom new()/delete() operators.
//Comment out this and '#include "allocator.cpp"' in support.cpp to use the original operators.
#include "allocator.h"

#include <stdint.h>
#include <thread>

#include "transaction.h"

#include "xthread.cpp"

atomic<int> slow_threads = 0;
atomic<int> objcnt = 0; //# of objects.
atomic<long> total = 0; //The sum of payloads.

#define TRANSACTIONAL_STRICT_assert

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
	};
};

typedef Transactional::Snapshot<LongNode> Snapshot;
typedef Transactional::Transaction<LongNode> Transaction;

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

#define NUM_SLOW_THREADS 2
#define NUM_THREADS 6

void
start_routine(int th_no) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    printf("start\n");

	int lps = 5000000;
	bool wait = false;
	if(th_no < NUM_SLOW_THREADS) {
		++slow_threads;
		lps = 10;
		wait = true;
	}

	shared_ptr<LongNode> p1(LongNode::create<LongNode>());
	shared_ptr<LongNode> p2(LongNode::create<LongNode>());
	for(int i = 0; i < lps; i++) {
		if( !slow_threads)
			break;
		p1->insert(p2);
		if((i % 10) == 0) {
			gn2->insert(p2);
			gn2->swap(p2, gn3);
			gn1->insert(p1);
		}
        gn1->iterate_commit([=](Transaction &tr1){
            Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			tr1[gn3] = ctr1[gn3] + 1;
            Snapshot &str1(tr1);
			tr1[gn1] = str1[gn1] - 1;
			tr1[gn2] = str1[gn2] + 1;
			if((i % 10) == 0) {
				tr1[p2] = str1[p2] + 1;
			}
			if(wait) msecsleep(10);
//			printf("f");
        });
		trans(*gn3) += 1;
        gn4->iterate_commit([=](Transaction &tr1){
            tr1[gn4] = tr1[gn4] + 1;
			tr1[gn4] = tr1[gn4] - 1;
			if(wait) msecsleep(30);
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
			if(wait) msecsleep(60);
//			printf("f");
        });
		trans(*gn3) += -1;
		if((i % 10) == 0) {
			gn2->release(p2);
			gn1->release(p1);
		}
	}
	long y = ***p2;
	if(y != 0) {
		printf("Error! P2=%ld\n", y);
		abort();
	}
	else
		printf("finish no=%d\n", th_no);
	if(th_no < NUM_SLOW_THREADS) {
		--slow_threads;
	}
    return;
}

int
main(int argc, char **argv)
{
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    for(int k = 0; k < 1; k++) {
        gn1.reset(LongNode::create<LongNode>());
        gn2.reset(LongNode::create<LongNode>());
        gn3.reset(LongNode::create<LongNode>());
        gn4.reset(LongNode::create<LongNode>());

		gn1->insert(gn2);
		Snapshot shot1(*gn2);
		gn2->insert(gn3);
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
		gn1->insert(p1);
		gn1->swap(p1, gn2);
		gn3->insert(p1);
		trans(*gn1) = 3;
		trans(*gn1) = 0;

		{
			shared_ptr<LongNode> p2(LongNode::create<LongNode>());
			shared_ptr<LongNode> p21(LongNode::create<LongNode>());
			shared_ptr<LongNode> p22(LongNode::create<LongNode>());
			shared_ptr<LongNode> p211(LongNode::create<LongNode>());
			p2->insert(p21);
			p21->insert(p211);
			p2->print_();
			p21->print_();
			p211->print_();
			p2->insert(p211);
			p21->insert(p22);
			p211->insert(p22);
			{
				Snapshot shot1(*p21);
				shot1[ *p21];
				shot1[ *p22];
				shot1[ *p211];
			}
			long x = ***p2;
			trans(*p22) = 1;
			trans(*p22) = 0;
			trans(*p21) = 1;
			trans(*p21) = 0;
			trans(*p211) = 1;
			trans(*p211) = 0;
			trans(*p2) = 1;
			trans(*p2) = 0;

            gn1->iterate_commit([=](Transaction &tr1){
                Snapshot &ctr1(tr1); // For reading.
				tr1[gn1] = ctr1[gn1] + 1;
				tr1[gn3] = ctr1[gn3] + 1;
                Snapshot &str1(tr1);
				tr1[gn1] = str1[gn1] - 1;
				tr1[gn3] = str1[gn3] - 1;
//				printf("f");
            });
		}
		gn1->release(p1);

        std::thread threads[NUM_THREADS];

        for(int i = 0; i < NUM_THREADS; i++) {
            std::thread th( &start_routine, i);
            threads[i].swap(th);
        }
		{
            msecsleep(1);
			gn3->insert(gn4);
            msecsleep(1);
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
