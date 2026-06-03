# kamestm/tests/tests.pri — STANDALONE test harness for the kamestm
# subtree mirror (one-way `git subtree split` from KAME).  Used only
# when the monorepo's ../../tests/tests.pri is absent (i.e. this repo
# was published as a leaf, not built inside kame).
#
# Self-contained: no kamepoolalloc, no kame, no Qt — just kamestm's
# headers + the Qt-free `support_standalone.h` shim alongside this
# .pri.  Binaries fall back to the system allocator transparently
# (kamestm calls zero pool-allocator runtime symbols; without
# libkamepoolalloc loaded, no global `operator new` override is in
# scope).
#
# Paths are anchored at THIS .pri's directory ($$PWD = kamestm/tests/)
# so they stay valid regardless of where each per-test .pro file lives.

TEMPLATE = app

CONFIG += exceptions
CONFIG += rtti
contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

CONFIG += c++17
CONFIG += console
CONFIG += testcase
CONFIG -= app_bundle
QT -= gui core
CONFIG -= qt

# kamestm headers live one level up; support_standalone.h is alongside.
INCLUDEPATH += $$PWD/..
INCLUDEPATH += $$PWD

# Preinclude the Qt-free `support.h` replacement.  Blocks the kame-only
# `support.h` (Qt + DEBUG_XTHREAD machinery) and provides minimal
# `XTime` / `msecsleep` / `timeStamp()` stubs.
QMAKE_CXXFLAGS += -include $$PWD/support_standalone.h
QMAKE_CXXFLAGS += -Wno-register

# Inline-compile the non-template kamestm TUs each test would otherwise
# have to link from libkamestm.  Cheaper than building libkamestm just
# to link it back in — and avoids a build-order dependency for the
# standalone repo where there is no SUBDIRS orchestration.
SOURCES += $$PWD/support_standalone.cpp
SOURCES += $$PWD/../threadlocal.cpp

HEADERS += $$PWD/support_standalone.h

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
