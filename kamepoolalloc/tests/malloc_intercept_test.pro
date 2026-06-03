TARGET = malloc_intercept_test

# Validate that §31 IAT redirect (KAMEPOOLALLOC_FULL_INTERCEPT, default-ON
# on Windows) routes plain stdlib malloc/calloc/realloc/free through the pool.
# Probe: kame_pool_malloc_usable_size returns 0 for foreign pointers;
# if intercept is active every malloc result comes back as a pool pointer.
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
    malloc_intercept_test.c
