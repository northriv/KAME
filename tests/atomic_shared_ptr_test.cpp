//#define msecsleep(x) (x)

//#include "xtime.h"

#include "support_standalone.h"

#include <stdint.h>
#include <thread>

#include "atomic_smart_ptr.h"
#include "xthread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;
atomic<int> xxx = 0;

//class A : public atomic_countable<A> {
class A {
public:
	A(long x) : m_x(x) {
//		fprintf(stdout, "c", x);
       ++objcnt;
       total += x;
       if(x < 0)
       		fprintf(stderr, " ??%d", (int)m_x);
	}
	virtual ~A() {
//		fprintf(stdout, "d", m_x);
		--objcnt;
       total -= m_x;
       assert(objcnt >= 0);
       if(total < 0)
       		fprintf(stderr, " ?%d,%d", (int)m_x, (int)total);
	}
    virtual long x() const {return m_x;}

long m_x;
};
class B : public A {
public:
    B(long x) : A(x) {
        ++objcnt;
//        fprintf(stdout, "C");
    }
    ~B() {
		--objcnt;
//        fprintf(stdout, "D");
    }
    virtual long x() const {return -m_x;}
    virtual long xorg() const {return m_x;}
};


atomic_shared_ptr<A> gp1, gp2, gp3;

void
start_routine() {
	printf("start\n");
	for(int i = 0; i < 400000; i++) {
    	local_shared_ptr<A> p1(new A(4));
    	assert(p1);
    	assert(p1.use_count() == 1);
    	local_shared_ptr<A> p2(new B(9));
    	local_shared_ptr<A> p3;
    	assert(!p3);

    	p2.swap(gp1);
    	gp2.reset(new A(32));
    	gp3.reset(new A(1001));

    	gp3.reset();
        gp1 = std::move(p2);
        assert( !p2);
    	p2 = gp1;
    	gp1 = gp1;
    	p2.swap(p3);

    	// 4 CAS variants selected by i % 4 (against gp1):
    	//   0: compareAndSet      (strong, const oldr)
    	//   1: compareAndSetWeak  (weak,   const oldr)
    	//   2: compareAndSwap     (strong, mutable oldr — updates on mismatch)
    	//   3: compareAndSetWeak  (weak,   scoped_atomic_view; ADAPTIVE)
    	switch(i % 4) {
    	case 0:
        	for(;;) {
        		local_shared_ptr<A> p(gp1);
        		if(p) xxx = p->x();
            	if(gp1.compareAndSet(p, p1)) break;
        	}
        	break;
    	case 1:
        	for(;;) {
        		local_shared_ptr<A> p(gp1);
        		if(p) xxx = p->x();
            	if(gp1.compareAndSetWeak(p, p1)) break;
        	}
        	break;
    	case 2:
        	for(local_shared_ptr<A> p(gp1);;) {
        		if(p) xxx = p->x();
            	if(gp1.compareAndSwap(p, p1)) break;
        	}
        	break;
    	case 3:
        	for(;;) {
            	scoped_atomic_view<A> sp(gp1,
            		scoped_atomic_view<A>::ADAPTIVE_THRESHOLD);
        		if(sp) xxx = sp->x();
            	if(gp1.compareAndSetWeak(sp, p1)) break;
        	}
        	break;
    	}
    	// Always exercise compareAndSwap on gp3 every iteration.
    	for(local_shared_ptr<A> p(gp3);;) {
    		if(p)
    			xxx = p->x();
	    	if(gp3.compareAndSwap(p, p1)) {
	    		break;
	    	}
//    		printf("f");
    	}
	}
	printf("finish\n");
    return;
}

#define NUM_THREADS 4

//! Basic local_weak_ptr sanity: construct from local_shared_ptr,
//! lock() while alive, expired() after the last strong drops, CB
//! deleted only when both counters hit zero.
static void test_local_weak_ptr_basic() {
    printf("test_local_weak_ptr_basic\n");
    int objcnt_before = objcnt;
    {
        local_shared_ptr<A> sp(new A(42));
        assert(objcnt == objcnt_before + 1);
        local_weak_ptr<A> wp(sp);
        assert( !wp.expired());
        {
            local_shared_ptr<A> locked = wp.lock();
            assert(locked);
            assert(locked->x() == 42);
            assert(sp.use_count() == 2);
        }
        assert(sp.use_count() == 1);
        sp.reset();
        //!< Object is destroyed (strong=0) but CB lives for the weak.
        assert(objcnt == objcnt_before);
        assert(wp.expired());
        assert( !wp.lock());
    }
    //!< wp out of scope → CB freed (no leak detection here, but the
    //!< Refcnt asserts in ~atomic_shared_ptr_gref_ catch double-free /
    //!< stuck refs).
}

//! Concurrent lock() racing strong destruction: stresses try_promote
//! CAS loop + release_strong_zero's fast-path branch.
static void test_local_weak_ptr_race() {
    printf("test_local_weak_ptr_race\n");
    constexpr int N = 4;
    constexpr int ITERS = 10000;
    int objcnt_before = objcnt;
    std::thread ths[N];
    for(int t = 0; t < N; ++t) {
        ths[t] = std::thread([](){
            for(int i = 0; i < ITERS; ++i) {
                local_shared_ptr<A> sp(new A(i + 1));
                local_weak_ptr<A> wp(sp);
                {
                    auto l = wp.lock();
                    assert(l && l->x() == i + 1);
                }
                sp.reset();
                assert(wp.expired());
            }
        });
    }
    for(auto &t : ths) t.join();
    assert(objcnt == objcnt_before);
}

int
main(int argc, char **argv)
{
    test_local_weak_ptr_basic();
    test_local_weak_ptr_race();

    std::thread threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++) {
        std::thread th( &start_routine);
        threads[i].swap(th);
    }
    for(int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
	printf("join\n");
	gp1.reset();
	gp2.reset();
	gp3.reset();
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
