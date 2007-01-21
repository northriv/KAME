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

atomic_queue<int, (SIZE + 1) * NUM_THREADS> queue1;
atomic_pointer_queue<int, (SIZE + 1) * NUM_THREADS> queue2;

void *
start_routine(void *) {
        usleep(100);
    for(int i = 0; i < SIZE; i++) {
        queue1.push(i);
        queue2.push(new int(i));
        usleep(1);
	}
}


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

    for(int i =0; i < SIZE * NUM_THREADS; i++) {
    	while(queue1.empty()) usleep(1);
        int x = queue1.front();
        queue1.pop();
    	while(queue2.empty()) usleep(1);
        int *t = queue2.front();
        queue2.pop();
        delete t;
        usleep(1);
    }

	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}    

    if(!queue1.empty() || !queue2.empty()) {
    	printf("failed\n");
    	return -1;
    }
	printf("succeeded\n");

	return 0;
}
