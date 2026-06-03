TARGET = malloc_intercept_test

# Validate that the strong-symbol / interpose / §31-IAT override layer routes
# the bare libc malloc family AND C++ operator new through the pool.  Probe:
# kame_pool_malloc_usable_size returns 0 for foreign pointers and >= the
# requested size for pool pointers.  The C-malloc assertions self-gate on a
# runtime probe (musl emits no strong malloc); operator-new / cross-route-free
# / kame_pool_* assertions run unconditionally.
#
# Same conditional logic as alloc_stress_test.pro — see that file for the
# rationale behind the exists()/!win32-*g++ branch.

exists(../../tests/tests.pri):!win32-*g++ {
    # ---- Monorepo (inside kame), non-MinGW ----
    include(../../tests/tests.pri)
} else {
    # ---- Standalone repo OR MinGW Windows monorepo ----
    TEMPLATE = app
    CONFIG += console c++17 exceptions rtti testcase
    CONFIG -= app_bundle qt
    QT -= core gui

    INCLUDEPATH += ..
    DEFINES += KAMEPOOLALLOC_DYLIB
    QMAKE_CXXFLAGS += -Wno-register

    contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
        CONFIG += sse sse2
    }
    win32-msvc* {
        QMAKE_CXXFLAGS += /arch:SSE2
    } else {
        contains(QMAKE_HOST.arch, x86) {
            QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
        }
    }

    SOURCES += ../allocator.cpp
    HEADERS += \
        ../allocator.h \
        ../allocator_prv.h \
        ../atomic_mfence.h \
        ../kame_pool.h
}

SOURCES += \
    malloc_intercept_test.cpp
