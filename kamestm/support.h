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
#ifndef supportH
#define supportH

// kamestm/support.h — minimal Qt-free `support.h` for downstream
// consumers that link `libkamestm` without the rest of KAME.
//
// This file defines only the symbol-visibility macros that the public
// headers (transaction.h, threadlocal.h, xtime.h, etc.) annotate
// declarations with.  It deliberately does NOT redefine `XTime`,
// `msecsleep`, or any other API surface — `xtime.h` etc. own those.
//
// Include-path priority that selects the right `support.h`:
//
//   * KAME application build (kame/kame.pro):
//       kame/ comes first on INCLUDEPATH -> kame/support.h
//       (Qt-aware, full version).
//
//   * Standalone test build (kamestm/tests/, kamepoolalloc/tests/):
//       The Qt-free harness in `kamestm/tests/support_standalone.h`
//       is preincluded via `-include`; it sets `supportH` and
//       provides the test-time `XTime` / `msecsleep` inline.
//       Any subsequent `#include "support.h"` lands on the shim in
//       `kamestm/tests/support.h`, which is a no-op (`supportH`
//       already defined).
//
//   * libkamestm dylib build (kamestm/kamestm.pro), or any
//     downstream consumer linking only `libkamestm` + `libkamepoolalloc`:
//       kame/ is NOT on the include path; this file (kamestm/support.h)
//       is found instead.  Provides DECLSPEC_KAME / DEBUG_XTHREAD /
//       a few similar macros — just enough to satisfy the header
//       declarations.  The real `xtime.h` defines `XTime` as usual,
//       and `xtime.cpp`'s out-of-line definitions match.

#ifndef DECLSPEC_KAME
    #define DECLSPEC_KAME
#endif
#ifndef DECLSPEC_MODULE
    #define DECLSPEC_MODULE
#endif
#ifndef DECLSPEC_SHARED
    #define DECLSPEC_SHARED
#endif

#include <cassert>
#ifdef NDEBUG
#  define DEBUG_XTHREAD 0
#else
#  define DEBUG_XTHREAD 1
#endif

#endif /* supportH */
