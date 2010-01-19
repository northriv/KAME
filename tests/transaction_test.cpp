/*
 * transaction_test.cpp
 *
 *  Created on: 2010/01/10
 *      Author: northriv
 */

#include "support.h"

#include <stdint.h>

#include "transaction.h"
#include <atomic.h>

#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;


class LongNode : public Node {
public:
	LongNode() {
		initPayload(new Payload(*this));
		++objcnt;
	}
	virtual ~LongNode() {
		--objcnt;
	}

	//! Data holder.
	struct Payload : public Node::Payload {
		Payload(Node &node) : Node::Payload(node), m_x(0) {}
		Payload(const Payload &x) : Node::Payload(x), m_x(x.m_x) {
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
	private:
		long m_x;
	};
};

shared_ptr<LongNode> gn1, gn2, gn3, gn4;

void *
start_routine(void *) {
	printf("start\n");
	shared_ptr<LongNode> p1(new LongNode);
	shared_ptr<LongNode> p2(new LongNode);
	for(int i = 0; i < 1000; i++) {
		if((i % 10) == 0)
			gn2->insert(p2);
		for(Transaction tr1(*gn1); ; ++tr1){
			Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			Snapshot str1(tr1);
			tr1[gn1] = str1[gn1] + 1;
			tr1[gn3] = str1[gn3] + 1;
			if((i % 10) == 0)
				tr1[p2] = str1[p2] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
		for(Transaction tr1(*gn2); ; ++tr1){
			Snapshot str1(tr1);
			tr1[gn2] = str1[gn2] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
		if((i % 10) == 0)
			gn2->release(p2);
		for(Transaction tr1(*gn4); ; ++tr1){
			Snapshot str1(tr1);
			tr1[gn4] = str1[gn4] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
	}
	long y = **gn1;
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

    for(int k = 0; k < 100; k++) {
		gn1.reset(new LongNode);
		gn2.reset(new LongNode);
		gn3.reset(new LongNode);
		gn4.reset(new LongNode);

		gn1->insert(gn2);
		gn2->insert(gn3);
		{
			Snapshot shot1(*gn1);
			shot1.print();
			long x = shot1[*gn3];
			printf("%ld\n", x);
		}

		shared_ptr<LongNode> p1(new LongNode);
		gn1->insert(p1);
		for(Transaction tr1(*gn1); ; ++tr1){
			Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			Snapshot str1(tr1);
			tr1[gn1] = str1[gn1] + 1;
			tr1[gn3] = str1[gn3] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
		gn1->release(p1);

	pthread_t threads[NUM_THREADS];
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_create(&threads[i], NULL, start_routine, NULL);
		}
		{
			usleep(1000);
			gn3->insert(gn4);
			usleep(1000);
			gn3->release(gn4);
		}
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_join(threads[i], NULL);
		}
		printf("join\n");
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
		printf("succeeded\n");
    }
	return 0;
}
