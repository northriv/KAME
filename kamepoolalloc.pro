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
    atomic_mfence.h \
    atomic_prv_mfence.h \
    atomic_prv_mfence_arm8.h \
    atomic_prv_mfence_x86.h

macx {
    # Plant the install_name so test binaries resolve us at runtime
    # via the `-Wl,-rpath,@executable_path/../kamepoolalloc` they
    # set in tests.pri.
    QMAKE_LFLAGS += -install_name @rpath/libkamepoolalloc.dylib
}

DESTDIR = $$OUT_PWD
