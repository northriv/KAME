#ifndef ATOMIC_CASN_H_
#define ATOMIC_CASN_H_

#include "atomic.h"
//! Restricted Version of Double CAS.
//! This descripter must be on stack.
//! restrict: clear bit0 of values.
class atomicRDCSS
{
public:
    //! \return true if succeeded.
    bool rdcss();
    intptr_t *target0;
    intptr_t *target1;
    intptr_t oldv0;
    intptr_t oldv1;
    intptr_t newv0;
    static intptr_t read(intptr_t *addr);
private:
    bool complete();
    bool isDesc(intptr_t v) {return (v & 1u);}
};
bool
atomicRDCSS::rdcss() {
    oldv0 &= ~1u;
    newv0 &= ~1u;
    oldv1 &= ~1u;
    for(;;) {
        if(atomicCompareAndSet((intptr_t)oldv0, ((intptr_t)this) | 1u, target0)) {
            return complete();
        }
        intptr_t r = *target0;
        if(!isDesc(r)) return false;
        ((atomicRDCSS *)(r & ~1u))->complete();
    }
}
bool
atomicRDCSS::complete() {
    if(oldv1 == *target1) {
        return atomicCompareAndSet(((intptr_t)this) | 1u, (intptr_t)newv0, target0);
    }
    else {
        atomicCompareAndSet(((intptr_t)this) | 1u, (intptr_t)oldv0, target0);    
        return false;
    }
}
intptr_t
atomicRDCSS::read(intptr_t *addr) {
    for(;;) {
        intptr_t r = *addr;
        if(!isDesc(r)) return r;
        ((atomicRDCSS *)(r & ~1u))->complete();
    }
}


//! This descripter must be on stack.
//! restrict: clear bit0 of values.
template <unsigned int n>
class atomicCASN {
    enum Status {UNDECIDED, SUCCEEDED, FAILED};
public:
    atomicCASN() : status(UNDECIDED) {}
    //! \return true if succeeded.
    bool compareAndSwap();
    intptr_t *target[n];
    intptr_t oldv[n];
    intptr_t newv[n];
    static intptr_t read(intptr_t *addr);
private:
    intptr_t status;
    bool isDesc(intptr_t v) {return (v & 1u);}
};

template <unsigned int n>
bool
atomicCASN<n>::compareAndSwap() {
    for(unsigned int i = 0; i < n; i++) {
        oldv[u] &= ~1u;
        newv[u] &= ~1u;
    }
    if(status == UNDECIDED) {
        intptr_t st  == SUCCEEDED;
        for(unsigned int i = 0; (i < n) && (st == SUCCEEDED); i++) {
            atomicRDCSS rdcss;
            rdcss.target0 = target[i];
            rdcss.oldv0 = oldv[i];
            rdcss.newv0 = ((intptr_t)this) | 1u;
            rdcss.target1 = &status;
            rdcss.oldv0 = UNDECIDED;
            if(!rdcss.rdcss()) {
                intptr_t val = atomicRDCSS::read(target[i]);
                if(isDesc(val)) {
                    ((atomicCASN *)(val & ~1u)))->compareAndSwap();
                    continue;
                }
                else
                    st = FAILED;
            }
        }
        atomicCompareAndSet(UNDECIDED, st, &status);
    }
    bool succeeded = (status == SUCCEEDED);
    for(unsigned int i = 0; i < n; i++) {
        atomicCompareAndSet(((intptr_t)this) | 1u, succeeded ? newv[i] : oldv[i], target[i]);
    }
    return succeeded;
}

template <unsigned int n>
intptr_t
atomicCASN<n>::read(intptr_t *addr) {
    for(;;) {
        intptr_t r = *addr;
        if(!isDesc(r)) return r;
        ((atomicCASN *)(r & ~1u))->compareAndSwap();
    }
}
#endif /*ATOMIC_CASN_H_*/
