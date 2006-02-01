#ifndef ATOMIC_H_
#define ATOMIC_H_

#include <stdint.h>

//! Lock-free synchronizations.

#if SIZEOF_VOID_P < SIZEOF_INT
 #error KAME assume assignment of int is atomic.
#endif

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
     //! memory barriers. 
     inline void readBarrier() {
            asm volatile( "" ::: "memory" );
     }
     inline void memoryBarrier() {
            asm volatile( "" ::: "memory" );
     }
     //! For spinning.
     inline void pauseN(unsigned int cnt) {
            for(unsigned int i = cnt; i != 0; --i) {
                asm volatile( "pause" ::: "memory" );
            }
     }
    //! \return true if old == *target and new value is assigned
    template <typename T>
    inline bool atomicCompareAndSet(T oldv, T newv, T *target ) {
       register unsigned char ret;
        asm volatile (
                "  lock; cmpxchg%z1 %1,%2;"
                " sete %0" // ret = zflag ? 1 : 0
                : "=q" (ret)
                : "r" (newv), "m" (*target), "a" (oldv)
                : "memory");
        return ret;
    }
    #if SIZEOF_VOID_P == 4
        #define HAVE_CAS_2
        //! Compare-And-Swap 2 long words.
        //! \param oldv0 compared with \p target[0].
        //! \param oldv1 compared with \p target[1].
        //! \param newv0 new value to \p target[0].
        //! \param newv1 new value to \p target[1].
        inline bool atomicCompareAndSet2(
            uint32_t oldv0, uint32_t oldv1,
            uint32_t newv0, uint32_t newv1, uint32_t *target ) {
           unsigned char ret;
            asm volatile (
                    " mov %6, %%ebx;"
                    "  lock; cmpxchg8b %7;"
                    " sete %0" // ret = zflag ? 1 : 0
                    : "=&b" (ret), "=&d" (oldv1), "=&a" (oldv0)
                    : "1" (oldv1), "2" (oldv0),
                     "c" (newv1), "r" (newv0),
                     "m" (*target)
                    : "memory");
            return ret;
        }
        inline bool atomicCompareAndSet2(
            int32_t oldv0, int32_t oldv1,
            int32_t newv0, int32_t newv1, int32_t *target ) {
               return atomicCompareAndSet2((uint32_t) oldv0, (uint32_t) oldv1,
                            (uint32_t) newv0, (uint32_t) newv1, (uint32_t*) target);
        }
    #endif
    template <typename T>
    inline T atomicSwap(T v, T *target ) {
        asm volatile (
                "xchg%z0 %0,%1" //lock prefix is not needed.
                : "=r" (v)
                : "m" (*target), "0" (v)
                : "memory" );
        return v;
    }
    template <typename T>
    inline void atomicAdd(T *target, T x ) {
        asm volatile (
                "lock; add%z0 %1,%0"
                :
                : "m" (*target), "ir" (x)
                : "memory" );
    }
    //! \return true if new value is zero.
    template <typename T>
    inline bool atomicAddAndTest(T *target, T x ) {
        register unsigned char ret;
        asm volatile (
                "lock; add%z1 %2,%1;"
                " sete %0" // ret = zflag ? 1 : 0
                : "=q" (ret)
                : "m" (*target), "ir" (x)
                : "memory" );
        return ret;
    }
    template <typename T>
    inline void atomicInc(T *target ) {
        asm volatile (
                "lock; inc%z0 %0"
                :
                : "m" (*target)
                : "memory" );
    }
    template <typename T>
    inline void atomicDec(T *target ) {
        asm volatile (
                "lock; dec%z0 %0"
                :
                : "m" (*target)
                : "memory" );
    }
    //! \return zero flag.
    template <typename T>
    inline bool atomicDecAndTest(T *target ) {
        register unsigned char ret;
        asm volatile (
                "lock; dec%z1 %1;"
                " sete %0" // ret = zflag ? 1 : 0
                : "=q" (ret)
                : "m" (*target)
                : "memory" );
        return ret;
    }
#else
    #if defined __ppc__ || defined __POWERPC__ || defined __powerpc__
         //! memory barriers. 
         inline void readBarrier() {
            asm volatile( "isync" ::: "memory" );
         }
         inline void memoryBarrier() {
            asm volatile( "sync" ::: "memory" );
         }
         //! For spinning.
         inline void pauseN(unsigned int cnt) {
                for(unsigned int i = cnt; i != 0; --i)
                    asm volatile( "nop" ::: "memory" );
         }
         template <typename T>
        //! \return true if old == *target and new value is assigned
         inline bool atomicCompareAndSet(T oldv, T newv, T *target ) {
            T ret;
            asm volatile ( "1: \n"
                    "lwarx %[ret], 0, %[target] \n"
                    "cmpw %[ret], %[oldv] \n"
                    "bne- 2f \n"
                    "stwcx. %[newv], 0, %[target] \n"
                    "bne- 1b \n"
                    "2: "
                    : [ret] "=&r" (ret)
                    : [oldv] "r" (oldv), [newv] "r" (newv), [target] "r" (target)
                    : "cc", "memory");
            return (ret == oldv);
         }
         //! \return target's old value.
         template <typename T>
         inline T atomicSwap(T newv, T *target ) {
            T ret;
            asm volatile ( "1: \n"
                    "lwarx %[ret], 0, %[target] \n"
                    "stwcx. %[newv], 0, %[target] \n"
                    "bne- 1b"
                    : [ret] "=&r" (ret)
                    : [newv] "r" (newv), [target] "r" (target)
                    : "cc", "memory");
            return ret;
         }
        template <typename T>
        inline void atomicInc(T *target ) {
            T ret;
            asm volatile ( "1: \n"
                    "lwarx %[ret], 0, %[target] \n"
                    "addi %[ret], %[ret], 1 \n"
                    "stwcx. %[ret], 0, %[target] \n"
                    "bne- 1b"
                    : [ret] "=&b" (ret)
                    : [target] "r" (target)
                    : "cc", "memory");
        }
        template <typename T>
        inline void atomicDec(T *target ) {
            T ret;
            asm volatile ( "1: \n"
                    "lwarx %[ret], 0, %[target] \n"
                    "addi %[ret], %[ret], -1 \n"
                    "stwcx. %[ret], 0, %[target] \n"
                    "bne- 1b"
                    : [ret] "=&b" (ret)
                    : [target] "r" (target)
                    : "cc", "memory");
        }
        template <typename T>
        inline void atomicAdd(T *target, T x ) {
            T ret;
            asm volatile ( "1: \n"
                    " lwarx %[ret], 0, %[target] \n"
                    "add %[ret], %[ret], %[x] \n"
                    "stwcx. %[ret], 0, %[target] \n"
                    "bne- 1b"
                    : [ret] "=&r" (ret)
                    : [target] "r" (target), [x] "r" (x)
                    : "cc", "memory");
        }
        //! \return true if new value is zero.
        template <typename T>
        inline bool atomicAddAndTest(T *target, T x ) {
            T ret;
            asm volatile ( "1: \n"
                    "lwarx %[ret], 0, %[target] \n"
                    "add %[ret], %[ret], %[x] \n"
                    "stwcx. %[ret], 0, %[target] \n"
                    "bne- 1b"
                    : [ret] "=&r" (ret)
                    : [target] "r" (target), [x] "r" (x)
                    : "cc", "memory");
            return (ret == 0);
        }
        //! \return zero flag.
        template <typename T>
        inline bool atomicDecAndTest(T *target ) {
            return atomicAddAndTest(target, (T)-1);
        }
   #else
        #error Unsupported processor
   #endif // __ppc__
#endif // __i386__

template <typename T>
class atomic
{
 public:
    atomic() : m_var(0) {}
    atomic(T t) : m_var(t) {}
    atomic(const atomic &t) : m_var(t) {}
    ~atomic() {}
    operator T() const {readBarrier(); return m_var;}
    atomic &operator=(T t) {m_var = t; readBarrier(); return *this;}
    atomic &operator++() {atomicInc(&m_var); return *this;}
    atomic &operator--() {atomicDecAndTest(&m_var); return *this;}
    atomic &operator+=(T t) {atomicAdd(&m_var, t); return *this;}
    atomic &operator-=(T t) {atomicAdd(&m_var, -t); return *this;}
    static T swap(T newv, atomic &t) {
        T old = atomicSwap(newv, &t.m_var);
        readBarrier();
        return old;
    }
    bool decAndTest() {
        bool ret = atomicDecAndTest(&m_var);
        readBarrier();
        return ret;
    }
    bool addAndTest(T t) {
        bool ret = atomicAddAndTest(&m_var, t);
        readBarrier();
        return ret;
    }
 private:
    T m_var;
};

#endif /*ATOMIC_H_*/
