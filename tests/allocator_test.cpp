/*
 * allocator_test.cpp
 */

#include "support.h"
#include "allocator.h" //lock-free custom new()/delete(). Comment this out to use the original operators.
#include <stdint.h>

#include "thread.cpp"
#include <deque>

atomic<int> objcnt = 0;
atomic<long> total = 0;

//class A : public atomic_countable {
class A {
public:
	A(long x) : m_x(x) {
//		fprintf(stdout, "c", x);
       ++objcnt;
       total += x;
       if(x < 0)
       		fprintf(stderr, " ??%d", (int)m_x);
	}
	virtual ~A() {
//		fprintf(stdout, "d", m_x);
		--objcnt;
       total -= m_x;
       ASSERT(objcnt >= 0);
       if(total < 0)
       		fprintf(stderr, " ?%d,%d", (int)m_x, (int)total);
	}
    virtual long x() const {return m_x;}

    long m_x;
};
class B : public A {
public:
	typedef long *plong;
    B(long x) : A(x) {
        ++objcnt;
//        fprintf(stdout, "C");
        arr = new plong[10];
        for(int i = 0; i < 10; ++i)
        	arr[i] = new long;
    }
    ~B() {
		--objcnt;
        for(int i = 0; i < 10; ++i)
        	delete arr[i];
		delete [] arr;
//        fprintf(stdout, "D");
    }
    virtual long x() const {return -m_x;}
    virtual long xorg() const {return m_x;}
    plong *arr;
};

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 100; i++) {
		A *a = new A(2);
		A *b = new B(4);
		delete a;
		delete b;
		std::deque<shared_ptr<A> > list;
		for(int j = 0; j < 10000; j++) {
			list.push_back(shared_ptr<A>(new A(1)));
			list.push_back(shared_ptr<A>(new B(1)));
		}
		list.clear();
	}
	printf("finish\n");
    return 0;
}

#define NUM_THREADS 4

int
main(int argc, char **argv) {
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

	A *a = new A(2);
	A *b = new B(4);
	delete a;
	delete b;
    for(int k = 0; k < 1; k++) {
	pthread_t threads[NUM_THREADS];
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_create(&threads[i], NULL, start_routine, NULL);
		}
		for(int i = 0; i < NUM_THREADS; i++) {
			pthread_join(threads[i], NULL);
		}
		printf("join\n");

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
