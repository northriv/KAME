/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef KAMEPOOLALLOC_ATOMIC_MFENCE_H_
#define KAMEPOOLALLOC_ATOMIC_MFENCE_H_

// Unified barriers + spin-pause.  See the in-repo
// atomic_prv_mfence.h (this directory) for full notes.  Replaces the
// previous arch-dispatched include of atomic_prv_mfence_x86.h /
// atomic_prv_mfence_arm8.h.  `kamepoolalloc` keeps its own copy
// (rather than depending on kame/) since it is built as a standalone
// dylib with no upward dependency on kame/.
#include "atomic_prv_mfence.h"

#endif /*KAMEPOOLALLOC_ATOMIC_MFENCE_H_*/
