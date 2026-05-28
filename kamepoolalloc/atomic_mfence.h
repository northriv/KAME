/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        implied.  See the License for the specific language governing
        permissions and limitations under the License.

        SPDX-License-Identifier: Apache-2.0
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
