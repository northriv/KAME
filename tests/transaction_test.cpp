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


class DoubleNode : public Node {
public:
	DoubleNode() {
		initPayload(new Payload(*this));
		++objcnt;
	}
	virtual ~DoubleNode() {
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

shared_ptr<DoubleNode> gn1;
shared_ptr<DoubleNode> gn2;
shared_ptr<DoubleNode> gn3;

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 1000; i++) {
		for(Transaction tr1(*gn1); ; ++tr1){
			Snapshot &ctr1(tr1); // For reading.
			tr1[gn1] = ctr1[gn1] + 1;
			Snapshot str1(tr1);
			tr1[gn1] = str1[gn1] + 1;
//			tr1[gn3] = str1[gn3] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
		for(Transaction tr1(*gn2); ; ++tr1){
			Snapshot str1(tr1);
			tr1[gn2] = str1[gn2] + 1;
			if(tr1.commit()) break;
			printf("f");
		}
	}
	long y = **gn1;
	printf("finish\n");
    return 0;
}

#define NUM_THREADS 5

int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    for(int k = 0; k < 100; k++) {
		gn1.reset(new DoubleNode);
		gn2.reset(new DoubleNode);
		gn3.reset(new DoubleNode);

		gn1->insert(gn2);
		gn2->insert(gn3);

	pthread_t threads[NUM_THREADS];
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_create(&threads[i], NULL, start_routine, NULL);
		}
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_join(threads[i], NULL);
		}
		printf("join\n");
		gn1.reset();
		gn2.reset();
		gn3.reset();

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
