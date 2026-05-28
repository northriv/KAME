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

INCLUDEPATH += $${_PRO_FILE_PWD_}/../kame
INCLUDEPATH += $${_PRO_FILE_PWD_}/../kamepoolalloc
INCLUDEPATH += $${_PRO_FILE_PWD_}/../kamestm

# Activate the dylib-mode `KAMEPOOLALLOC_DYLIB` codepath in
# `allocator.h`: `activateAllocator()` collapses to an inline no-op
# and `KamePooledAllocGuard` becomes a true no-op — the dylib's own
# `__attribute__((constructor))` flips `g_sys_image_loaded = true`
# at load, before any consumer image's static init.
DEFINES += KAMEPOOLALLOC_DYLIB

# Preinclude standalone support header to block Qt-dependent originals
QMAKE_CXXFLAGS += -include $${_PRO_FILE_PWD_}/support_standalone.h

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
SOURCES += support_standalone.cpp
SOURCES += ../kamestm/threadlocal.cpp

# Link against libkamepoolalloc — built by ../kamepoolalloc/kamepoolalloc.pro
# in the sibling subdir (top-level kame.pro orders this before tests via
# `tests.depends = kamepoolalloc`).  The path is the per-test build dir's
# `../kamepoolalloc` (each per-test .pro builds into its own subdir under
# build/.../tests, parallel to build/.../kamepoolalloc).
LIBS += -L$$OUT_PWD/../kamepoolalloc -lkamepoolalloc

macx {
    # rpath: search for libkamepoolalloc.dylib relative to the test
    # binary's directory.  `@executable_path/../kamepoolalloc/` covers
    # the `build/.../tests/<test>` → `build/.../kamepoolalloc/` hop.
    QMAKE_LFLAGS += -Wl,-rpath,@executable_path/../kamepoolalloc
}
unix:!macx {
    QMAKE_LFLAGS += -Wl,-rpath,\\$\$ORIGIN/../kamepoolalloc
}

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
HEADERS += support_standalone.h
HEADERS += ../kamepoolalloc/allocator.h
HEADERS += ../kamepoolalloc/allocator_prv.h
HEADERS += ../kamepoolalloc/atomic_mfence.h
HEADERS += ../kamepoolalloc/atomic_prv_mfence.h
HEADERS += ../kamepoolalloc/atomic_prv_mfence_x86.h
HEADERS += ../kamepoolalloc/atomic_prv_mfence_arm8.h
HEADERS += ../kame/threadlocal.h
HEADERS += ../kame/atomic.h
HEADERS += ../kame/atomic_prv_basic.h
HEADERS += ../kame/atomic_prv_std.h
HEADERS += ../kame/atomic_prv_mfence.h
HEADERS += ../kame/atomic_prv_mfence_x86.h
HEADERS += ../kame/atomic_prv_mfence_arm8.h

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
