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

# Preinclude standalone support header to block Qt-dependent originals
QMAKE_CXXFLAGS += -include $${_PRO_FILE_PWD_}/support_standalone.h

# Common sources brought into every standalone test binary:
#  - support_standalone.cpp: Qt-free stub for `support.cpp` / `xtime.cpp`
#  - tests/allocator.cpp:    activator wrapper that #includes
#                            ../kame/allocator.cpp (operator new/delete
#                            overrides + activateAllocator() in a static
#                            ctor).  On non-x86 the pool is auto-disabled
#                            via USE_STD_ALLOCATOR (see ../kame/allocator.h)
#                            and this file contributes nothing.
#  - ../kame/threadlocal.cpp: `detail::tls_storage()` — the type-erased
#                            TLS dispatcher every `XThreadLocal<T, Tag>`
#                            calls into; lives in kame.dll for the real
#                            build, must be linked into the standalone
#                            test binary directly.
SOURCES += $${_PRO_FILE_PWD_}/support_standalone.cpp
SOURCES += $${_PRO_FILE_PWD_}/allocator.cpp
SOURCES += $${_PRO_FILE_PWD_}/../kame/threadlocal.cpp

# Headers that every standalone test transitively pulls in via
# `support_standalone.h` / `allocator.h` / `threadlocal.h`.  Listed
# here so Qt Creator's project navigator shows them, and so renames /
# cross-references in those headers are picked up by the IDE for every
# test target.
HEADERS += $${_PRO_FILE_PWD_}/support_standalone.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/allocator.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/allocator_prv.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/threadlocal.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/atomic.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/atomic_prv_basic.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/atomic_prv_std.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/atomic_prv_mfence_x86.h
HEADERS += $${_PRO_FILE_PWD_}/../kame/atomic_prv_mfence_arm8.h

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
