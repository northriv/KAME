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

int
main(int argc, char **argv)
{
    timeval tv;
    gettimeofday(&tv, 0);
    srand(tv.tv_usec);

    #define SIZE 10
    for(int i = 0; i < 10; i++) {
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
    for(int i = 0; i < 10; i++) {
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
    
	return 0;
}
