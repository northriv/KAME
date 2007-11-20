#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

#include <stdint.h>

#include "atomic.h"
#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<int> total = 0;
atomic<double> test = 1.0;

class A {
public:
	A(int x) : m_a(x * 2.0), m_b(x * 3.0) {
//		fprintf(stdout, "c", x);
       ++objcnt;
       total += x;
       test = x;
	}
	virtual ~A() {
//		fprintf(stdout, "d", m_x);
		--objcnt;
		double x = m_b / 6.0;
       total -= lrint(m_a / 2.0 + x * 2.0) / 2;
       double a = test;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 2.0) > 1e-40)) total -= 1;
       if(fabs(m_a * 3.0 - m_b * 2.0) > 1e-30) total -= 1;
    }
	
	atomic<double> m_a;
	char m_dummy;
	atomic<double> m_b;
};

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 100000; i++) {
    	scoped_ptr<A> p1(new A(1));
    	scoped_ptr<A> p2(new A(2));
	}
	printf("finish\n");
    return 0;
}

#define NUM_THREADS 8

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
