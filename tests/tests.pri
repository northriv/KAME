TEMPLATE = app

CONFIG += exceptions
CONFIG += rtti
contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

CONFIG += c++17

CONFIG += console
CONFIG += testcase
CONFIG -= app_bundle #macosx

# Standalone tests use `support_standalone.{h,cpp}` instead of the real
# `support.h` / `xtime.cpp` — Qt-free, C++17 stdlib only.  Disable Qt
# linkage so qmake doesn't pull QtCore into the link line.  The few
# tests that were declaring this themselves can drop the duplicate.
QT -= gui core
CONFIG -= qt

# Paths are relative to the directory containing THIS .pri file
# ($$PWD = git/tests/), not the per-test .pro file's directory —
# the latter now varies (kamepoolalloc/tests/, kamestm/tests/) so
# $${_PRO_FILE_PWD_} would give different roots and break.
#
# Deliberately NO ../kame on the path: kamestm and kamepoolalloc sit
# BELOW kame in the dependency stack.  The standalone test harness
# replaces `support.h` / `xtime.h` via the preinclude below, so kame/
# never needs to be on the include path.
INCLUDEPATH += $$PWD/../kamepoolalloc
INCLUDEPATH += $$PWD/../kamestm
INCLUDEPATH += $$PWD/../kamestm/tests

# Activate the dylib-mode `KAMEPOOLALLOC_DYLIB` codepath in
# `allocator.h`: `activateAllocator()` collapses to an inline no-op
# and `KamePooledAllocGuard` becomes a true no-op — the dylib's own
# `__attribute__((constructor))` flips `g_sys_image_loaded = true`
# at load, before any consumer image's static init.
DEFINES += KAMEPOOLALLOC_DYLIB

# Preinclude standalone support header to block Qt-dependent originals
QMAKE_CXXFLAGS += -include $$PWD/../kamestm/tests/support_standalone.h

# Common sources brought into every standalone test binary:
#  - support_standalone.cpp:    Qt-free stub for `support.cpp` / `xtime.cpp`
#  - ../kame/threadlocal.cpp:   `detail::tls_storage()` — the type-erased
#                               TLS dispatcher every `XThreadLocal<T,
#                               Tag>` calls into; lives in kame.dll for
#                               the real build, must be linked into the
#                               standalone test binary directly.
#
# Notes on what is NOT compiled inline anymore:
#  - tests/allocator.cpp:       gone.  The dylib auto-activates at load
#                               via `__attribute__((constructor))` so
#                               the static-init activator shim is
#                               obsolete.
#  - ../kamepoolalloc/allocator.cpp:
#                               built as a shared library by the
#                               sibling `kamepoolalloc.pro` (added to
#                               the top-level kame.pro SUBDIRS and
#                               `tests.depends`).  Tests `LIBS` against
#                               it via the macx/unix-specific block at
#                               the bottom of this file.
SOURCES += $$PWD/../kamestm/tests/support_standalone.cpp
SOURCES += $$PWD/../kamestm/threadlocal.cpp

# Link against the two dylibs that own the STM + allocator machinery:
# `libkamepoolalloc` (kamepoolalloc/kamepoolalloc.pro) and `libkamestm`
# (kamestm/kamestm.pro).  Build ORDER is already wired in the top-level
# kame.pro via `tests.depends = kamepoolalloc kamestm`, so both dylibs
# exist before any test links.
#
# Their build dirs are NOT a flat sibling of each per-test build dir:
# a test at `kamestm/tests/foo.pro` shadow-builds under
# `<build>/kamestm/tests/`, while the dylibs land at
# `<build>/kamepoolalloc` and `<build>/kamestm`.  So `$$OUT_PWD/../X`
# resolves to the wrong place (`<build>/kamestm/X`).  Anchor the lib
# paths at the build ROOT instead — derived from THIS .pri's source dir
# (`<src>/tests`) via `$$shadowed`, which maps source→shadow-build for
# the current .pro.  Correct regardless of how deep the test's OUT_PWD
# nests (and reduces to the source tree for an in-source build).
KAME_BUILD_ROOT = $$shadowed($$PWD/..)
LIBS += -L$$KAME_BUILD_ROOT/kamepoolalloc -lkamepoolalloc
LIBS += -L$$KAME_BUILD_ROOT/kamestm -lkamestm

# Absolute rpaths into the build tree so the in-place test binaries
# resolve both dylibs at runtime irrespective of their nesting depth
# (`@executable_path`-relative paths would need a per-layout `../`
# count).  `-Wl,-rpath,<abs>` is honoured by both ld64 and GNU ld.
QMAKE_LFLAGS += -Wl,-rpath,$$KAME_BUILD_ROOT/kamepoolalloc
QMAKE_LFLAGS += -Wl,-rpath,$$KAME_BUILD_ROOT/kamestm

# Headers that every standalone test transitively pulls in via
# `support_standalone.h` / `allocator.h` / `threadlocal.h`.  Listed
# here so Qt Creator's project navigator shows them, and so renames /
# cross-references in those headers are picked up by the IDE for every
# test target.
#
# Paths are relative to the per-test .pro file (i.e. tests/) — same
# convention as the SOURCES list above.  Plain relative paths (rather
# than the previous `$${_PRO_FILE_PWD_}/...` form) keep Qt Creator's
# project navigator from showing a stale `tests/..` segment in the
# resolved path display.
HEADERS += $$PWD/../kamestm/tests/support_standalone.h
HEADERS += $$PWD/../kamepoolalloc/allocator.h
HEADERS += $$PWD/../kamepoolalloc/allocator_prv.h
HEADERS += $$PWD/../kamepoolalloc/atomic_mfence.h
HEADERS += $$PWD/../kamestm/threadlocal.h
HEADERS += $$PWD/../kamestm/atomic.h

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
