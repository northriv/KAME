#define SIZEOF_INT 4
#define SIZEOF_LONG 4
#define SIZEOF_VOID_P 4
#define SIZEOF_SHORT 2
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

#include <stdint.h>


void my_assert(char const*s, int d) {
        fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

//#define HAVE_CAS_2
// fake cas2
inline bool atomicCompareAndSet2(
    uint32_t oldv0, uint32_t oldv1,
    uint32_t newv0, uint32_t newv1, uint32_t *target ) {
        ASSERT(oldv0 == target[0]);
        ASSERT(oldv1 == target[1]);
        if(rand() > RAND_MAX/2) {
            target[0] = newv0;
            target[1] = newv1;
            return true;
        }
        return false;
    }

#include "atomic_smart_ptr.h"
#include "thread.cpp"

int objcnt = 0;

class A {
public:
	A(int x) : m_x(x) {
		fprintf(stdout, "Created %d\n", x);
        atomicInc(&objcnt);
	}
	virtual ~A() {
		fprintf(stdout, "Destroyed %d\n", m_x);
        atomicDec(&objcnt);
	}
    virtual int x() const {return m_x;} 
	
int m_x;
};
class B : public A {
public:
    B(int x) : A(x) {
        fprintf(stdout, "Created B\n");
    }
    ~B() {
        fprintf(stdout, "Destroyed B\n");
    }
    virtual int x() const {return -m_x;} 
    virtual int xorg() const {return m_x;} 
};

int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    for(int i = 0; i < 10; i++) {
        	atomic_shared_ptr<A> p1(new A(1));
        	atomic_shared_ptr<A> p2(new B(2));
        	atomic_shared_ptr<A> p3;
        	atomic_shared_ptr<A> p4(p1);
        	atomic_shared_ptr<A> p5(p3);
        	atomic_shared_ptr<A> p6(new A(6));
        	p2.swap(p1);
        	p5.reset(new A(51));
        	p3.reset(new A(3));
        
        	p2.reset(new A(21));
        	p3 = p2;
        	p4 = p3;
        	p5.reset();
        	p6.swap(p3);
        atomic_shared_ptr<A> old(p4);
        ASSERT(p5.compareAndSwap(p3, p4) == false);
        	ASSERT(p5.compareAndSwap(old, p4));
        	p4.swap(p2);
        atomic_shared_ptr<const A> p4c(p4);
    }
    if(objcnt != 0) return -1;
	return 0;
}
