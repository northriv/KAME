#define msecsleep(x) (x)

//#include "xtime.h"

#include "support.h"

//For inline expansion of lock-free custom new()/delete() operators.
//Comment out this and '#include "allocator.cpp"' in support.cpp to use the original operators.
#include "allocator.h"

#include <stdint.h>
#include <thread>


//#define HAVE_CAS_2
// fake cas2
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

#include "atomic_smart_ptr.h"
#include "atomic_queue.h"
#include "xthread.cpp"


#define SIZE 100000
#define NUM_THREADS 4

atomic_queue<int, (SIZE + 100) * NUM_THREADS> queue1;
atomic_pointer_queue<int, NUM_THREADS - 1> queue2;
atomic_queue_reserved<int, NUM_THREADS-  1> queue3;
typedef atomic_queue_reserved<int, NUM_THREADS-  1>::key atomic_queue_reserved_key;

atomic<int> g_queue1_total = 0, g_queue2_total = 0, g_queue3_total = 0;
atomic<int> g_cnt = 0;

void
start_routine(void) {
    for(int j = 0; j < SIZE; j++) {
        int i;
        for(;;) {
            i = g_cnt;
            if(g_cnt.compare_exchange_strong(i, i+1)) break;
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
                int x = *t;
                if(queue2.atomicPop(t)) {
                    assert(x >= 0);
                    *t = -100;
                    g_queue2_total -= x;
//		        	break;
                }
            }
//	    	printf("2");
        }
       {// for(;;) {
        int x;
            atomic_queue_reserved_key key = queue3.atomicFront(&x);
            if(key) {
                if(queue3.atomicPop(key)) {
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
        int x;
            atomic_queue_reserved_key key =queue3.atomicFront(&x);
            if(key) {
                if(queue3.atomicPop(key)) {
                    g_queue3_total -= x;
        //        	break;
                }
            }
            //printf("3");
        }
    }

    if(!queue1.empty() || !queue2.empty() || !queue3.empty() ||
        (g_queue1_total != 0) || (g_queue3_total != 0) || (g_queue2_total != 0)) {
        printf("\ntest1:failed queue1size=%d, queue1total=%d, queue2size=%d, queue2total=%d, queue3size=%d, queue3total=%d\n",
            queue1.size(), (int)g_queue1_total,
            queue2.size(), (int)g_queue2_total,
           queue3.size(), (int)g_queue3_total);
    }

std::thread threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++) {
        std::thread th( &start_routine);
        threads[i].swap(th);
    }

    for(int i =0; i < SIZE * NUM_THREADS; i++) {
         if(queue1.empty()) continue;
        int x = queue1.front();
        g_queue1_total -= x;
        queue1.pop();
    }

    for(int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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
        printf("\ntest2:failed queue1size=%d, queue1total=%d, queue2size=%d, queue2total=%d, queue3size=%d, queue3total=%d\n",
            queue1.size(), (int)g_queue1_total,
            queue2.size(), (int)g_queue2_total,
           queue3.size(), (int)g_queue3_total);
        return -1;
    }
    else
        printf("succeeded\n");

    return 0;
}
