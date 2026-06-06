TARGET = kamepoolalloc
TEMPLATE = lib

# Shared library — required for `__DATA,__interpose` `free` redirection
# on macOS (dyld only processes interpose sections from MH_DYLIB images)
# and for the dylib's `__attribute__((constructor))` auto-activate to
# fire before any consumer image's static init.
#
# Production kame.app (kame/kame.pro) still inline-compiles
# `kamepoolalloc/allocator.cpp` — the dylib boundary buys us LTO-safe
# `free` interpose semantics that production (non-LTO) doesn't need.
# Tests link against this dylib in BOTH cmake and qmake builds, so the
# LTO behaviour is testable identically across both build systems.

# `CONFIG += plugin` mirrors the modules' build pattern and drops the
# trailing `1.0.0` version suffix that qmake's default `lib` template
# adds to dylibs — keeps the tests' link line and rpath simple
# (`-lkamepoolalloc` resolves to `libkamepoolalloc.dylib` /
#  `libkamepoolalloc.so` without dance around symlinks).
CONFIG += plugin
CONFIG -= qt
CONFIG -= app_bundle

CONFIG += c++17
QMAKE_CXXFLAGS += -Wno-register

# Hot-path codegen (see tests/CMakeLists.txt for the full rationale): hide
# the inline/COMDAT internal functions so intra-DSO self-calls bind directly
# instead of through the PLT — notably `kame_free`'s call to the inline
# `PoolAllocatorBase::deallocate` on every free.  `-fno-semantic-interposition`
# does the same for the non-inline self-calls.  Exported symbols are
# unchanged, so the `free`/`malloc` interpose semantics are untouched.  Not
# applicable to MSVC.
!win32-msvc*: QMAKE_CXXFLAGS += -fvisibility-inlines-hidden -fno-semantic-interposition

# Dylib-mode auto-activate + KamePooledAllocGuard / activateAllocator()
# elision (see allocator.cpp / allocator.h).  Consumers (tests) also
# define this so the header sees the same codepath — qmake has no
# cmake-style PUBLIC propagation, so the define is duplicated in
# `tests.pri`.
DEFINES += KAMEPOOLALLOC_DYLIB

contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

INCLUDEPATH += $${_PRO_FILE_PWD_}

SOURCES += allocator.cpp

HEADERS += \
    allocator.h \
    allocator_prv.h \
    atomic.h \
    atomic_mfence.h \
    atomic_smart_ptr.h \
    kame_pool.h

macx {
    # Plant the install_name so test binaries resolve us at runtime
    # via the `-Wl,-rpath,@executable_path/../kamepoolalloc` they
    # set in tests.pri.
    QMAKE_LFLAGS += -install_name @rpath/libkamepoolalloc.dylib
}

win32-*g++ {
    # MinGW gcc / clang via lld: export every symbol from the DLL and
    # plant the import lib next to it so consumers (kame.exe, the
    # tests) can link via `-lkamepoolalloc`.  Mirrors `kame/kame.pro`'s
    # win32-g++ block.  `-Wl,--export-all-symbols` is required for
    # lld — bfd ld silently auto-exported, lld does not, so symbols
    # like `kame_pool_set_realtime_mode` / `activateAllocator` would
    # otherwise come up undefined at consumer link time.
    QMAKE_LFLAGS += -Wl,--export-all-symbols -Wl,--out-implib,lib$${TARGET}.a
}
win32-msvc* {
    # /utf-8: source files contain UTF-8 box-drawing characters and Japanese
    # text in comments.  Without this flag MSVC reads files as the system
    # code page (CP932 on Japanese Windows), which misinterprets the last
    # byte of some UTF-8 sequences as a lead byte, potentially swallowing
    # the following newline and hiding preprocessor #define directives inside
    # comments.  Symptom: "error C2065: 'ALLOC_CHUNK_HEADER': undeclared
    # identifier" on lines that follow box-drawing comment blocks.
    QMAKE_CXXFLAGS += /utf-8
    # MSVC: every extern "C" / non-static C++ symbol needs an explicit
    # `__declspec(dllexport)` at definition time.  The pool's public
    # surface is annotated via a `DECLSPEC_KAMEPOOLALLOC` macro that
    # picks dllexport/dllimport from this define; without the
    # annotations the kame_pool_* C API isn't visible from the import
    # library.  (TODO: add DECLSPEC_KAMEPOOLALLOC annotations through
    # kame_pool.h / allocator.h to support the MSVC path; current
    # supported Windows toolchain is MinGW.)
    DEFINES += DECLSPEC_KAMEPOOLALLOC=__declspec(dllexport)
}

DESTDIR = $$OUT_PWD
