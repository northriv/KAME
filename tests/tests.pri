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

# Mirror source-tree paths into the build output so
# `tests/allocator.cpp` (activator shim) and
# `../kamepoolalloc/allocator.cpp` (the real pool TU) — both basename
# `allocator.cpp` — produce distinct .o files instead of clobbering
# each other in a flat OBJECTS_DIR.
CONFIG += object_parallel_to_source

# Preinclude standalone support header to block Qt-dependent originals
QMAKE_CXXFLAGS += -include $${_PRO_FILE_PWD_}/support_standalone.h

# Common sources brought into every standalone test binary:
#  - support_standalone.cpp:    Qt-free stub for `support.cpp` / `xtime.cpp`
#  - tests/allocator.cpp:       activator shim — declares
#                               `extern void activateAllocator()` and
#                               calls it from a static-init object so
#                               the pool is enabled before main().  No
#                               longer #includes the allocator TU itself.
#  - ../kamepoolalloc/allocator.cpp:
#                               the actual pool-allocator TU (was
#                               `kame/allocator.cpp` before the
#                               kamepoolalloc/ split).  Compiled inline
#                               into each test binary here in qmake; the
#                               cmake build builds it as a shared
#                               library (libkamepoolalloc.dylib) instead
#                               so dyld can process the `__DATA,
#                               __interpose` `free` redirection in it.
#                               On non-x86 the pool is auto-disabled via
#                               USE_STD_ALLOCATOR (see allocator.h) and
#                               this file contributes nothing.
#  - ../kame/threadlocal.cpp:   `detail::tls_storage()` — the type-erased
#                               TLS dispatcher every `XThreadLocal<T,
#                               Tag>` calls into; lives in kame.dll for
#                               the real build, must be linked into the
#                               standalone test binary directly.
SOURCES += support_standalone.cpp
SOURCES += allocator.cpp
SOURCES += ../kamepoolalloc/allocator.cpp
SOURCES += ../kame/threadlocal.cpp

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
HEADERS += ../kamepoolalloc/atomic_prv_mfence_x86.h
HEADERS += ../kamepoolalloc/atomic_prv_mfence_arm8.h
HEADERS += ../kame/threadlocal.h
HEADERS += ../kame/atomic.h
HEADERS += ../kame/atomic_prv_basic.h
HEADERS += ../kame/atomic_prv_std.h
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
