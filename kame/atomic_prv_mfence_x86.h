/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_MFENCE_X86_H_
#define ATOMIC_PRV_MFENCE_X86_H_

// Backward-compat shim.  The x86-specific intrinsic-based barriers
// have been folded into atomic_prv_mfence.h (std::atomic_thread_fence
// based, portable C++17).  Direct include of this header is
// discouraged — include atomic_prv_mfence.h instead.
#include "atomic_prv_mfence.h"

#endif /*ATOMIC_PRV_MFENCE_X86_H_*/
