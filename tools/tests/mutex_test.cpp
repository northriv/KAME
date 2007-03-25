#define msecsleep(x) (x)

//#include "xtime.h"

#include <support.h>

#include <stdint.h>

#include <thread.h>
#include <thread.h>
#include "thread.cpp"

atomic<int> objcnt = 0;
atomic<int> total = 0;

XMutex g_mutex;
XRecursiveMutex g_rec_mutex;
int g_cnt1 = 0;
int g_cnt2 = 0;

void *
start_routine(void *) {
	printf("start\n");
	for(int i = 0; i < 100000; i++) {
		XScopedLock<XMutex> lock(g_mutex);
		{
			XScopedLock<XRecursiveMutex> lock(g_rec_mutex);
			{
				XScopedLock<XRecursiveMutex> lock(g_rec_mutex);
				{
					XScopedLock<XRecursiveMutex> lock(g_rec_mutex);
					g_cnt2++;
				}
			}
			g_cnt2--;
		}
		g_cnt1++;
		writeBarrier();
		g_cnt1--;
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
    if(g_cnt1 != 0) {
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
