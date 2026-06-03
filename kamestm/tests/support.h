/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            ../LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see ../LICENSE-GPL-2.0).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/

/*
 * support.h — kamestm-test-side shim that redirects to the Qt-free
 * `support_standalone.h`.  When kamestm/tests/ is on the include path
 * (and kame/ is intentionally NOT — see kamestm/tests/CMakeLists.txt),
 * every `#include "support.h"` from kamestm/ headers resolves to this
 * file, which forwards to the standalone harness.
 *
 * The standalone harness uses the same `#ifndef supportH` guard as
 * kame/support.h, so a `-include support_standalone.h` preinclude
 * + this shim are interchangeable — both end with `supportH` defined
 * and the standalone definitions installed.
 *
 * In production KAME builds, kame/ comes first on the include path
 * and kame/support.h (the Qt-aware version) wins; this shim is never
 * reached.
 */

#ifndef supportH
#include "support_standalone.h"
#endif
