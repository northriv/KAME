#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

//For inline expansion of lock-free custom new()/delete() operators.
//Comment out this and '#include "allocator.cpp"' in support.cpp to use the original operators.
#include "allocator.h"

#include <stdint.h>

//
//#ifndef HAVE_CAS_2
//inline bool atomicCompareAndSet2(
//    uint32_t oldv0, uint32_t oldv1,
//    uint32_t newv0, uint32_t newv1, uint32_t *target ) {
//        assert(oldv0 == target[0]);
//        assert(oldv1 == target[1]);
//        if(rand() > RAND_MAX/2) {
//            target[0] = newv0;
//            target[1] = newv1;
//            return true;
//        }
//        return false;
//    }
//#endif

#include "atomic_smart_ptr.h"
#include "xthread.cpp"

atomic<int> objcnt = 0;

class A {
public:
	A(int x) : m_x(x) {
//		fprintf(stdout, "c", x);
        ++objcnt;
	}
	virtual ~A() {
//		fprintf(stdout, "d", m_x);
        --objcnt;
	}
    virtual int x() const {return m_x;}

int m_x;
};
class B : public A {
public:
    B(int x) : A(x) {
//        fprintf(stdout, "C");
    }
    ~B() {
//        fprintf(stdout, "D");
    }
    virtual int x() const {return -m_x;}
    virtual int xorg() const {return m_x;}
};


atomic_unique_ptr<A> gp1, gp2, gp3;

void
start_routine(void) {
	for(int i = 0; i < 1000000; i++) {
    	atomic_unique_ptr<A> p1(new A(1));
    	atomic_unique_ptr<A> p2(new B(2));
    	atomic_unique_ptr<A> p3;


    	p2.swap(gp1);
    	gp2.reset(new A(51));
    	gp3.reset(new A(3));

    	gp3.reset();
    	p2.swap(p3);

    	p2.reset();
    	p2.swap(gp1);
	}
}

#define NUM_THREADS 4

int
main(int argc, char **argv)
{
std::thread threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++) {
        std::thread th( &start_routine);
        threads[i].swap(th);
    }
    for(int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    gp1.reset();
	gp2.reset();
	gp3.reset();
    if(objcnt != 0) {
    	printf("failed\n");
    	return -1;
    }
	printf("succeeded\n");
	return 0;
}
