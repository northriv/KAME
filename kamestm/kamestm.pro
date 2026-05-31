TARGET = kamestm
TEMPLATE = lib

# Shared library carrying the STM framework's non-template TUs:
#
#   - threadlocal.cpp : type-erased TLS dispatcher (`detail::tls_storage()`)
#                       that every `XThreadLocal<T, Tag>` lands on.
#   - xthread.cpp     : portable thread-handle layer (QThread vs
#                       std::thread / pthread, switched by USE_QTHREAD).
#   - xtime.cpp       : monotonic time helpers used by the Lamport-clock
#                       serial numbers.
#
# Everything else in kamestm/ is template-heavy header code that is
# instantiated per consumer (Node<XN>, Snapshot<XN>, Transaction<XN>,
# atomic_shared_ptr<T>, etc.), so the .cpp footprint of the library is
# deliberately small.
#
# kamestm is self-contained: every header it needs (atomic_mfence.h,
# fast_vector.h) lives in this directory.  It does NOT call any
# libkamepoolalloc runtime symbol, so the standalone kamestm subtree
# mirror (one-way `git subtree split`) builds with nothing under
# kamepoolalloc/ present — when kamepoolalloc is absent, consumers run
# against the system allocator transparently (no global `operator new`
# override is in scope without libkamepoolalloc loaded).
#
# Production kame.app (kame/kame.pro) inline-compiles these same .cpp
# files (no DLL boundary) AND also inline-compiles kamepoolalloc's
# allocator.cpp; the pool override is wired up there independently of
# kamestm.  The standalone kamestm dylib exists for tests and for
# downstream consumers that want to embed kamestm without KAME.

CONFIG += plugin
CONFIG -= qt
CONFIG -= app_bundle

CONFIG += c++17
QMAKE_CXXFLAGS += -Wno-register

contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

INCLUDEPATH += $${_PRO_FILE_PWD_}

# Qt-free `support.h` is provided right alongside the headers
# (kamestm/support.h) — minimal macro shim, distinct from the
# test-only kamestm/tests/support.h shim which forwards to
# `support_standalone.h` (the latter would shadow `XTime` /
# `msecsleep` and break the dylib's xtime.cpp link).

SOURCES += \
    threadlocal.cpp \
    xthread.cpp \
    xtime.cpp

HEADERS += \
    atomic.h \
    atomic_mfence.h \
    atomic_queue.h \
    atomic_smart_ptr.h \
    fast_vector.h \
    support.h \
    threadlocal.h \
    transaction.h \
    transaction_detail.h \
    transaction_definitions.h \
    transaction_impl.h \
    transaction_negotiation.h \
    transaction_neg_impl.h \
    transaction_signal.h \
    xthread.h \
    xtime.h

macx {
    # rpath install_name so consumers can resolve via
    # `-Wl,-rpath,@executable_path/../kamestm`.
    QMAKE_LFLAGS += -install_name @rpath/libkamestm.dylib
}

# No link to libkamepoolalloc: kamestm calls zero libkamepoolalloc
# runtime symbols.  Binaries that want pool-allocate behaviour link
# libkamepoolalloc directly (tests/tests.pri / kame/kame.pro do so);
# binaries that don't run against the system allocator.

DESTDIR = $$OUT_PWD
