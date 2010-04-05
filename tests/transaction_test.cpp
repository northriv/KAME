/*
 * transaction_test.cpp
 *
 *  Created on: 2010/01/10
 *      Author: northriv
 */

#include "support.h"
//#include "allocator.h"
#include <stdint.h>

#include "transaction.h"
#include <atomic.h>

#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;

//#define TRANSACTIONAL_STRICT_ASSERT

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
		}
		Payload &operator+=(const long &x) {
			total += x;
			m_x += x;
		}
	private:
		long m_x;
	};
};

#define trans(node) for(Transaction \
	__implicit_tr(node, false); !__implicit_tr.isModified() || !__implicit_tr.commitOrNext(); ) __implicit_tr[node]

template <class T>
typename boost::enable_if<boost::is_base_of<LongNode, T>,
	const typename Transactional::SingleSnapshot<LongNode, T> >::type
 operator*(T &node) {
	return Transactional::SingleSnapshot<LongNode, T>(node);
}

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, gn2, gn3, gn4;

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 2500; i++) {
		//		gn1->_print();
		for(Transaction tr1(*gn1); ; ++tr1){
//			tr1.print();
			Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			tr1[gn3] = ctr1[gn3] + 1;
			Snapshot str1(tr1);
			tr1[gn1] = str1[gn1] - 1;
			tr1[gn2] = str1[gn2] + 1;
			if(tr1.commit()) break;
//			printf("f");
		}
		{
			Snapshot shot(*gn1);
			ASSERT(shot[*gn2] <= shot[*gn3]);
		}
		trans(*gn3) += 1;
		for(Transaction tr1(*gn4); ; ++tr1){
			tr1[gn4] = tr1[gn4] + 1;
			tr1[gn4] = tr1[gn4] - 1;
			if(tr1.commit()) break;
//			printf("f");
		}
		{
			Snapshot shot(*gn2);
			ASSERT(shot[*gn2] <= shot[*gn3]);
		}
		for(Transaction tr1(*gn2); ; ++tr1){
			Snapshot str1(tr1);
			tr1[gn2] = tr1[gn2] - 1;
			tr1[gn3] = str1[gn3] - 1;
			if(tr1.commit()) break;
//			printf("f");
		}
		trans(*gn3) += -1;
	}
	printf("finish\n");
    return 0;
}

#define NUM_THREADS 4

int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    for(int k = 0; k < 50; k++) {
		gn1.reset(LongNode::create<LongNode>());
		gn2.reset(LongNode::create<LongNode>());
		gn3.reset(LongNode::create<LongNode>());
		gn4.reset(LongNode::create<LongNode>());

		for(Transaction tr1(*gn1); ; ++tr1){
			if( !gn1->insert(tr1, gn2, true))
				continue;
			tr1[ *gn2] = tr1[ *gn2] + 1;
			if( !gn2->insert(tr1, gn3, true))
				continue;
			tr1.print();
			if( !gn3->insert(tr1, gn4, true))
				continue;
			tr1.print();
			if( !gn3->release(tr1, gn4))
				continue;
			tr1.print();
			if(tr1.commit())
				break;
		}
		gn1->_print();
		gn1->release(gn2);
		gn1->_print();
		gn1->insert(gn2);
		gn1->_print();
		for(Transaction tr1(*gn2); ; ++tr1){
			tr1[ *gn2] = tr1[ *gn2] - 1;
			tr1[ *gn3] = 0;
			if(tr1.commit())
				break;
		}
		{
			Snapshot shot1(*gn1);
			shot1.print();
			long x = shot1[*gn3];
			printf("Gn3:%ld\n", x);
		}
		trans(*gn3) = 3;
		long x = **gn3;
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
			shared_ptr<LongNode> p22(LongNode::create<LongNode>());
			shared_ptr<LongNode> p211(LongNode::create<LongNode>());
			shared_ptr<LongNode> p2111(LongNode::create<LongNode>());
			shared_ptr<LongNode> p2112(LongNode::create<LongNode>());
			shared_ptr<LongNode> p2113(LongNode::create<LongNode>());
			shared_ptr<LongNode> p2114(LongNode::create<LongNode>());
			p2111->insert(p2112);
			p2112->insert(p2113);
			p2111->insert(p2114);
			shared_ptr<LongNode> p21(LongNode::create<LongNode>());
			p2->insert(p21);
			p21->insert(p211);
			p2->insert(p211);
			trans(*p211) = 1;
			p21->insert(p22);
			p211->insert(p22);
			for(Transaction tr1(*gn3); ; ++tr1){
				if( !p1->insert(tr1, p22, true))
					continue;
				if( !gn3->insert(tr1, p2, true))
					continue;
				if( !gn3->insert(tr1, p2111, true))
					continue;
				if( !p21->insert(tr1, p2111, false))
					continue;
				tr1[*p22] = 1;
//				{ Snapshot shot1( *p211); shot1.list(); }
//				{ Snapshot shot1( *p1); shot1.list(); }
//				{ Snapshot shot1( *p2); shot1.list(); }
				if(tr1.commit()) break;
//				if(tr1.commitAt( *gn1)) break;
				printf("f");
			}
			{
				Snapshot shot1(*p21);
				shot1[ *p2111];
				shot1[ *p2112];
				shot1[ *p2113];
				shot1[ *p2114];
			}
			{
				Snapshot shot1(*gn3);
				shot1[ *p2];
				shot1[ *p21];
				shot1[ *p22];
				shot1[ *p2111];
				shot1[ *p2114];
			}
			trans(*p211) = 0;
			for(Transaction tr1(*gn3); ; ++tr1){
				tr1[ *p2113] = 1;
				tr1[ *p2114] = 1;
				if( !p1->release(tr1, p22))
					continue;
				if( !gn3->release(tr1, p2))
					continue;
				if( !gn3->release(tr1, p2111))
					continue;
				if(tr1.commit()) break;
				printf("f");
			}
			trans(*p22) = 0;
			p2114->_print();
			trans(*p2114) = 0;
			trans(*p2113) = 0;

			for(Transaction tr1(*gn1); ; ++tr1){
				Snapshot &ctr1(tr1); // For reading.
				tr1[gn1] = ctr1[gn1] + 1;
				tr1[gn3] = ctr1[gn3] + 1;
				Snapshot str1(tr1);
				tr1[gn1] = str1[gn1] - 1;
				tr1[gn3] = str1[gn3] - 1;
				if(tr1.commit()) break;
				printf("f");
			}
		}
		gn1->_print();
		gn1->release(p1);
		gn1->_print();

	pthread_t threads[NUM_THREADS];
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_create(&threads[i], NULL, start_routine, NULL);
		}
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_join(threads[i], NULL);
		}
		printf("join\n");

		if(**gn1 || **gn2 || **gn3 || **gn4) {
			printf("failed1\n");
			printf("Gn1:%ld\n", (long)**gn1);
			printf("Gn2:%ld\n", (long)**gn2);
			printf("Gn3:%ld\n", (long)**gn3);
			printf("Gn4:%ld\n", (long)**gn4);
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
