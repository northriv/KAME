/***************************************************************************
        Copyright (C) 2002-2021 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_MFENCE_ARM8_H_
#define ATOMIC_PRV_MFENCE_ARM8_H_

#ifdef _MSC_VER
    #error
#else

#endif

//! memory barriers.
inline void readBarrier() noexcept {
     __asm__ __volatile__ ("dmb" : : : "memory");
 //   __dmb(0xf);
}
inline void writeBarrier() noexcept {
     __asm__ __volatile__ ("dmb" : : : "memory");
 //   __dmb(0xf);
}
inline void memoryBarrier() noexcept {
     __asm__ __volatile__ ("dmb" : : : "memory");
 //   __dmb(0xf);
}

inline void pause4spin() noexcept {
    __asm__ __volatile__("isb" ::: "memory");
//    ___isb(0xf);
}

#endif /*ATOMIC_PRV_MFENCE_ARM8_H_*/
