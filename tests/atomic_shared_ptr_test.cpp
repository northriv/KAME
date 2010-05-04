//#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

//For inline expansion of lock-free custom new()/delete() operators.
//Comment out this and '#include "allocator.cpp"' in support.cpp to use the original operators.
#include "allocator.h"

#include <stdint.h>

#include "atomic_smart_ptr.h"
#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;
atomic<int> xxx = 0;

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
    B(long x) : A(x) {
        ++objcnt;
//        fprintf(stdout, "C");
    }
    ~B() {
		--objcnt;
//        fprintf(stdout, "D");
    }
    virtual long x() const {return -m_x;}
    virtual long xorg() const {return m_x;}
};


atomic_shared_ptr<A> gp1, gp2, gp3;

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 400000; i++) {
    	local_shared_ptr<A> p1(new A(4));
    	ASSERT(p1);
    	ASSERT(p1.use_count() == 1);
    	local_shared_ptr<A> p2(new B(9));
    	local_shared_ptr<A> p3;
    	ASSERT(!p3);

    	p2.swap(gp1);
    	gp2.reset(new A(32));
    	gp3.reset(new A(1001));

    	gp3.reset();
    	gp1 = p2;
    	p2 = gp1;
    	gp1 = gp1;
    	p2.swap(p3);

    	for(;;) {
    		local_shared_ptr<A> p(gp1);
    		if(p)
    			xxx = p->x();
	    	if(gp1.compareAndSet(p, p1)) {
	    		break;
	    	}
//    		printf("f");
    	}
    	for(local_shared_ptr<A> p(gp3);;) {
    		if(p)
    			xxx = p->x();
	    	if(gp3.compareAndSwap(p, p1)) {
	    		break;
	    	}
//    		printf("f");
    	}
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

pthread_t threads[NUM_THREADS];
	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_create(&threads[i], NULL, start_routine, NULL);
	}
	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	printf("join\n");
	gp1.reset();
	gp2.reset();
	gp3.reset();
	if(objcnt != 0) {
    	printf("failed\n");
    	return -1;
    }
    if(total != 0) {
    	printf("failed\n");
    	return -1;
    }
	printf("succeeded\n");
	return 0;
}
