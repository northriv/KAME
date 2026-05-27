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
#ifndef KAME_ALLOCATOR_H_SHIM_
#define KAME_ALLOCATOR_H_SHIM_

// Backward-compatible shim.  The pool-allocator headers and translation
// unit moved into `kamepoolalloc/` (see project root) so that the
// allocator can be built as its own shared library (`libkamepoolalloc`)
// — necessary for `__DATA,__interpose`-based `free` redirection on
// macOS, since dyld only honours interposing from MH_DYLIB images.
//
// Existing call sites such as `kame/transaction_signal.h` keep their
// `#include "allocator.h"` line; this shim relays to the new location.
// Build systems must add `kamepoolalloc/` to the include path:
//   * cmake (tests):  `include_directories(.../kamepoolalloc)`
//   * qmake (prod):   `INCLUDEPATH += $$PWD/../kamepoolalloc`
#include "../kamepoolalloc/allocator.h"

#endif /*KAME_ALLOCATOR_H_SHIM_*/
