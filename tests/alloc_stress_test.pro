TARGET = alloc_stress_test

# kamepoolalloc lives in two layouts: inside the kame monorepo (as a
# subtree) and as a standalone GitHub mirror (one-way `git subtree split`).
# kame/tests/tests.pri sits ABOVE this subtree, so it only exists in the
# monorepo.  Branch on its presence so the same .pro builds in both.
exists(../../tests/tests.pri):!win32 {
    # ---- Monorepo (inside kame), non-Windows ----
    # Delegate to kame's shared standalone-test harness: it links the
    # prebuilt libkamepoolalloc + libkamestm dylibs (built by the sibling
    # .pro files via kame.pro's SUBDIRS) and pulls in the Qt-free support
    # stubs.  Byte-identical to before the standalone decoupling — the
    # kamestm bits are vestigial for the allocator tests but harmless here
    # (kamestm is built anyway in the monorepo).
    #
    # Windows is excluded: PE/COFF can't export `operator new` from
    # libkamepoolalloc.dll, and the linker binds the test's new/delete to
    # libc++'s defaults regardless of link order — so the pool would never
    # be exercised (no "Reserve swap space"; symptom #2 of the MinGW
    # handoff).  On Windows we fall through to the inline-compile branch
    # below, which gives this self-contained allocator test its OWN pool
    # `operator new` (exactly as kame.exe does).
    include(../../tests/tests.pri)
} else {
    # ---- Standalone kamepoolalloc repo, OR Windows monorepo ----
    # tests.pri (and kamestm) are absent (standalone) or skipped (Windows),
    # so the test must be self-contained.  Inline-compile allocator.cpp with
    # KAMEPOOLALLOC_DYLIB defined: its `__attribute__((constructor))`
    # auto-activates the pool at binary startup — the same effect the dylib
    # has in the monorepo / cmake builds — so a single .pro needs no
    # SUBDIRS/rpath/link-order orchestration and the test needs no explicit
    # activateAllocator() call.  This is a pure pool test (no kamestm, no
    # cross-module frees), so it needs no free-redirect.  The dylib boundary
    # itself (LTO-safe `free` interpose) stays exercised by the cmake
    # standalone build (tests/CMakeLists.txt) and the non-Win qmake build.
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
    alloc_stress_test.cpp
