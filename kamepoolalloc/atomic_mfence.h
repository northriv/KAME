/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
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
