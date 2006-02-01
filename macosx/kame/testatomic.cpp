//!g++  -g -I/sw/include/qt -I../../kame -I/sw/include -I.. testatomic.cpp -o testatomic 

#define SIZEOF_INT 4
#define SIZEOF_LONG 4
#define SIZEOF_VOID_P 4
#define SIZEOF_SHORT 2
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define __BIG_ENDIAN__
#define MACOSX
#define msecsleep(x) (x)

//#include "xtime.h"
//#include "support.h"
#include "atomic.h"

void my_assert(bool x, char const*s, int d) {
	if(!x)
		fprintf(stderr, "Err:%s:%d\n", s, d);
}


class A {
public:
	A(int x) : m_x(x) {
		fprintf(stderr, "Created %d\n", x);
	}
	~A() {
		fprintf(stderr, "Destroyed %d\n", m_x);
	}
	
int m_x;
};

int
main()
{
	atomic_shared_ptr<A> p1(new A(10));
	atomic_shared_ptr<A> p2(new A(11));
	atomic_shared_ptr<A> p3;
	atomic_shared_ptr<A> p4(p1);
	atomic_shared_ptr<A> p5(p3);
	atomic_shared_ptr<A> p6(new A(15));
	
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
//	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p2.reset(new A(111));
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
//	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p3 = p2;
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p4 = p3;
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p5.reset();
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p6.swap(p3);
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
//	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p4.swap(p5);
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
//	fprintf(stderr, "p4 %d\n", (*p4).m_x);
	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	p4.swap(p2);
	fprintf(stderr, "p1 %d\n", (*p1).m_x);
//	fprintf(stderr, "p2 %d\n", (*p2).m_x);
	fprintf(stderr, "p3 %d\n", (*p3).m_x);
	fprintf(stderr, "p4 %d\n", (*p4).m_x);
	fprintf(stderr, "p5 %d\n", (*p5).m_x);
	fprintf(stderr, "p6 %d\n", (*p6).m_x);
	
	return 0;
}