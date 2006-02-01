//!g++  -g -I/sw/include/qt -I../../kame -I/sw/include -I.. testatomic.cpp -o testatomic 

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
//#define HAVE_CAS_2
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
#include "atomic_queue.h"
#include "thread.cpp"

#define a_shared_ptr shared_ptr
#define a_scoped_ptr scoped_ptr
#define a_shared_ptr atomic_shared_ptr
#define a_scoped_ptr atomic_scoped_ptr

void my_assert(char const*s, int d) {
		fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

class A {
public:
	A(int x) : m_x(x) {
		fprintf(stderr, "Created %d\n", x);
	}
	virtual ~A() {
		fprintf(stderr, "Destroyed %d\n", m_x);
	}
    virtual int x() const {return m_x;} 
	
int m_x;
};
class B : public A {
public:
    B(int x) : A(x) {
        fprintf(stderr, "Created %d\n", x);
    }
    ~B() {
        fprintf(stderr, "Destroyed %d\n", m_x);
    }
    virtual int x() const {return -m_x;} 
    virtual int xorg() const {return m_x;} 
};

int
main(int argc, char **argv)
{
    srand(atoi(argv[1]));
	a_shared_ptr<A> p1(new A(10));
	a_shared_ptr<A> p2(new B(11));
	a_shared_ptr<A> p3;
	a_shared_ptr<A> p4(p1);
	a_shared_ptr<A> p5(p3);
	a_shared_ptr<A> p6(new A(15));
	a_scoped_ptr<A> p10(new A(20));
	a_scoped_ptr<A> p11(new A(21));
	a_scoped_ptr<A> p12(new A(22));
	p12.reset();
	p10.swap(p11);
	p12.reset(new A(23));
	p12.reset(new A(24));
	
    atomic<int> a = 3;
    ASSERT(a == 3);
    a += 1;
    ASSERT(a == 4);
    
    #define SIZE 10
    {
        atomic_queue<int, SIZE + 1> queue;
        for(int j = 0; j < SIZE; j++) {
            ASSERT(queue.empty());
            for(int i =0; i < SIZE; i++)
                queue.push(i);
            for(int i =0; i < SIZE/2; i++) {
                ASSERT(!queue.empty());
                int t = queue.front();
                ASSERT(t == i);
                queue.pop();
            }
            for(int i =0; i < SIZE/2; i++)
                queue.push(i + SIZE);
            for(int i =0; i < SIZE; i++) {
                ASSERT(!queue.empty());
                int t = queue.front();
                ASSERT(t == i + SIZE/2);
                queue.pop();
            }
            ASSERT(queue.empty());
        }
    }
    {
        atomic_pointer_queue<int, SIZE + 1> queue;
        for(int j = 0; j < SIZE; j++) {
            ASSERT(queue.empty());
            for(int i =0; i < SIZE; i++)
                queue.push(new int(i));
            for(int i =0; i < SIZE; i++) {
                ASSERT(!queue.empty());
                int *t = queue.front();
                ASSERT(*t == i);
                queue.pop();
                delete t;
            }
            ASSERT(queue.empty());
        }
    }
    
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
//	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p2.reset(new A(111));
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
//	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p3 = p2;
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p4 = p3;
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p5.reset();
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p6.swap(p3);
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
//	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p4.swap(p5);
	fprintf(stderr, "p1 %d\n", (*p1).x());
	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
//	fprintf(stderr, "p4 %d\n", (*p4).x());
	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	p4.swap(p2);
	fprintf(stderr, "p1 %d\n", (*p1).x());
//	fprintf(stderr, "p2 %d\n", (*p2).x());
	fprintf(stderr, "p3 %d\n", (*p3).x());
	fprintf(stderr, "p4 %d\n", (*p4).x());
	fprintf(stderr, "p5 %d\n", (*p5).x());
	fprintf(stderr, "p6 %d\n", (*p6).x());
	new A(99);
	return 0;
}