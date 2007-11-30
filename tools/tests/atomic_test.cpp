#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

#include <stdint.h>

#include "atomic.h"
#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<int> total = 0;
//typedef atomic<double> a_double;
typedef volatile double a_double;
struct Test {
	a_double x1;
	char d1;
	a_double x2;
	char d1a;
	a_double x2a;
	char d1b;
	a_double x2b;
	char d2;
	a_double x3;
	char d3;
	a_double x4;
} test = {1.0, 1, 1.0, 1, 1.0, 1, 1.0, 1, 1.0, 1, 1.0};

class A {
public:
	A(int x) : m_a(x * 2.0), m_b(x * 3.0) {
//		fprintf(stdout, "c", x);
	   test.x1 = x;
       ++objcnt;
       total += x;
       test.x2 = x;
       test.x3 = x;
       test.x4 = x;
       double a = test.x1;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x2;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x3;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x4;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
	}
	virtual ~A() {
//		fprintf(stdout, "d", m_x);
		--objcnt;
		double x = m_b / 6.0;
       total -= lrint(m_a / 2.0 + x * 2.0) / 2;
       double a = test.x1;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x2;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x3;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
       a = test.x4;
       if((fabs(a - 1.0) > 1e-40) && (fabs(a - 40.0) > 1e-40)) total -= 1;
//       if(fabs(m_a * 3.0 - m_b * 2.0) > 1e-30) total -= 1;
    }
	
	atomic<double> m_a;
	int m_dummy;
	atomic<double> m_b;
};

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 100000; i++) {
    	A p1(1), p2(40), p3(1), p4(40);
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
