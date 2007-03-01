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
// fake cas2
//inline bool atomicCompareAndSet2(
//    uint32_t oldv0, uint32_t oldv1,
//    uint32_t newv0, uint32_t newv1, uint32_t *target ) {
//        ASSERT(oldv0 == target[0]);
//        ASSERT(oldv1 == target[1]);
//        if(rand() > RAND_MAX/2) {
//            target[0] = newv0;
//            target[1] = newv1;
//            return true;
//        }
//        return false;
//    }

#include "atomic_smart_ptr.h"
#include "atomic_queue.h"
#include "thread.cpp"

void my_assert(char const*s, int d) {
		fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

#define SIZE 10000
#define NUM_THREADS 8

atomic_queue<int, (SIZE + 100) * NUM_THREADS> queue1;
atomic_pointer_queue<int, NUM_THREADS> queue2;
atomic_queue_reserved<int, NUM_THREADS> queue3;

atomic<int> g_queue2_total = 0, g_queue3_total = 0;
atomic<int> g_cnt = 0;

void *
start_routine(void *) {
	try {
    usleep(100);
    for(int j = 0; j < SIZE; j++) {
    	int i;
    	for(;;) {
	    	i = g_cnt;
	    	if(g_cnt.compareAndSet(i, i+1)) break;
    	}
    	
        queue1.push(i);
        queue3.push(i);
        queue2.push(new int(i));
        g_queue2_total += i;
        g_queue3_total += i;
        usleep(1);
        for(;;) {
	        int *t = (int*)queue2.atomicFront();
	        if(t) {
		        int x = *t;
		        if(queue2.atomicPop(t)) {
			        ASSERT(x >= 0);
		        	*t = -100;
		        	g_queue2_total -= x;
		        	break;
		        }
	        }
	    	printf("2");
        }
        for(;;) {
	        int *t = (int*)queue3.atomicFront();
	        if(t) {
		        int x = *t;
		        ASSERT(x >= 0);
		        if(queue3.atomicPop(t)) {
		        	g_queue3_total -= x;
		        	break;
		        }
	        }
	    	printf("3");
        }
	}
	}
	catch (...) {
		printf("ahoh\n");
	}
}


int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    for(int i = 0; i < NUM_THREADS; i++) {
        queue3.push(i);
        queue2.push(new int(i));
        g_queue2_total += i;
        g_queue3_total += i;
    }
    for(int i = 0; i < NUM_THREADS; i++) {
        for(;;) {
	        const int *t =queue2.atomicFront();
	        if(t) {
		        const int x = *t;
		        if(queue2.atomicPop(t)) {
		        	g_queue2_total -= x;
		        	break;
		        }
	        }
	    	printf("2");
        }
        for(;;) {
	        const int *t =queue3.atomicFront();
	        if(t) {
		        const int x = *t;
		        if(queue3.atomicPop(t)) {
		        	g_queue3_total -= x;
		        	break;
		        }
	        }
	    	printf("3");
        }
	}

pthread_t threads[NUM_THREADS];
	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_create(&threads[i], NULL, start_routine, NULL);
	}

	int64_t total = 0;
    for(int i =0; i < SIZE * NUM_THREADS; i++) {
    	while(queue1.empty()) usleep(1);
        int x = queue1.front();
        total += x;
        queue1.pop();
    }

	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}    

    if(!queue1.empty() || !queue2.empty() || !queue3.empty() || 
    	(total != ((int64_t)SIZE* NUM_THREADS)*(SIZE* NUM_THREADS-1)/2) ||
    	(g_queue3_total != 0) || (g_queue2_total != 0)) {
    	printf("failed total=%lld, cal=%lld, queue2size=%d, queue2total=%d, queue3size=%d, queue3total=%d\n", 
    		total,((int64_t)SIZE* NUM_THREADS)*(SIZE* NUM_THREADS-1)/2,
			queue2.size(), (int)g_queue2_total, queue3.size(), (int)g_queue3_total);
    }
    else
		printf("succeeded\n");

	return 0;
}
