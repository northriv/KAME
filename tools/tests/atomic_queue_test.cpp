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

#define SIZE 100000
#define NUM_THREADS 8

atomic_queue<int, (SIZE + 100) * NUM_THREADS> queue1;
atomic_pointer_queue<int, NUM_THREADS -1> queue2;
atomic_queue_reserved<int, NUM_THREADS- 1> queue3;

atomic<int> g_queue1_total = 0, g_queue2_total = 0, g_queue3_total = 0;
atomic<int> g_cnt = 0;

void *
start_routine(void *) {
    usleep(100);
    for(int j = 0; j < SIZE; j++) {
    	int i;
    	for(;;) {
	    	i = g_cnt;
	    	if(g_cnt.compareAndSet(i, i+1)) break;
    	}
    	
        try {
	        queue1.push(i);
	        g_queue1_total += i;
        }
        catch (...) {
			printf("1");
        }
        try {
	        queue3.push(i);
	        g_queue3_total += i;
        }
        catch (...) {
			printf("3");
        }
        try {
	        queue2.push(new int(i));
	        g_queue2_total += i;
        }
        catch (...) {
			printf("2");
        }
       {// for(;;) {
	        int *t = (int*)queue2.atomicFront();
	        if(t) {
	        	ASSERT(t);
		        int x = *t;
		        if(queue2.atomicPop(t)) {
			        ASSERT(x >= 0);
		        	*t = -100;
		        	g_queue2_total -= x;
//		        	break;
		        }
	        }
//	    	printf("2");
        }
       {// for(;;) {
	        int *t = (int*)queue3.atomicFront();
	        if(t) {
		        int x = *t;
		        if(queue3.atomicPop(t)) {
		        	g_queue3_total -= x;
//		        	break;
		        }
	        }
//	    	printf("3");
        }
	}
}


int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    for(int i = 0; i < NUM_THREADS; i++) {
        try {
	        queue3.push(i);
	        g_queue3_total += i;
        }
        catch (...) {
			printf("3");
        }
        try {
	        queue2.push(new int(i));
	        g_queue2_total += i;
        }
        catch (...) {
			printf("2");
        }
    }
    for(int i = 0; i < NUM_THREADS; i++) {
       {// for(;;) {
	        const int *t =queue2.atomicFront();
	        if(t) {
		        const int x = *t;
		        if(queue2.atomicPop(t)) {
		        	g_queue2_total -= x;
//		        	break;
		        }
	        }
	//    	printf("2");
        }
       {// for(;;) {
	        const int *t =queue3.atomicFront();
	        if(t) {
		        const int x = *t;
		        if(queue3.atomicPop(t)) {
		        	g_queue3_total -= x;
		//        	break;
		        }
	        }
	    	//printf("3");
        }
	}

pthread_t threads[NUM_THREADS];
	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_create(&threads[i], NULL, start_routine, NULL);
	}

    for(int i =0; i < SIZE * NUM_THREADS; i++) {
    	 if(queue1.empty()) usleep(1);
    	 if(queue1.empty()) continue;
        int x = queue1.front();
        g_queue1_total -= x;
        queue1.pop();
    }

	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}    

    for(;;) {
    	 if(queue1.empty()) break;
        int x = queue1.front();
        g_queue1_total -= x;
        queue1.pop();
    }
    	
    for(;;) {
    	 if(queue2.empty()) break;
        int *x = queue2.front();
        g_queue2_total -= *x;
        queue2.pop();
    }
    for(;;) {
    	 if(queue3.empty()) break;
        int x = queue3.front();
        g_queue3_total -= x;
        queue3.pop();
    }

       
    if(!queue1.empty() || !queue2.empty() || !queue3.empty() || 
    	(g_queue1_total != 0) || (g_queue3_total != 0) || (g_queue2_total != 0)) {
    	printf("failed queue1size=%d, queue1total=%d, queue2size=%d, queue2total=%d, queue3size=%d, queue3total=%d\n", 
			queue1.size(), (int)g_queue1_total,
			queue2.size(), (int)g_queue2_total,
		   queue3.size(), (int)g_queue3_total);
    }
    else
		printf("succeeded\n");

	return 0;
}
