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
# (§36b) The lock-free atomic primitives (atomic.h / atomic_smart_ptr.h /
# atomic_mfence.h) were RELOCATED to kamepoolalloc/ so they have a single home
# and a single GenMC/TLA verification, shared by both the allocator and kamestm
# (the allocator's §36b orphan reuse uses atomic_shared_ptr).  kamestm now
# sources these HEADERS from kamepoolalloc/ (added to INCLUDEPATH below) — they
# are header-only, so this is a COMPILE-time include dependency, NOT a link
# dependency: kamestm still calls no libkamepoolalloc runtime symbol.  The
# standalone kamestm subtree-split therefore needs kamepoolalloc/'s atomic
# headers present (header-only); when libkamepoolalloc is not LOADED, consumers
# still run against the system allocator transparently (no global `operator
# new` override is in scope without libkamepoolalloc loaded).
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
# (§36b) atomic.h / atomic_smart_ptr.h / atomic_mfence.h were relocated to
# kamepoolalloc/ so the lock-free atomic primitives have a single home (and a
# single GenMC/TLA verification) shared by both the allocator and kamestm.
# kamestm now sources them from there.  Standalone kamestm therefore depends on
# kamepoolalloc's atomic headers (header-only; no libkamepoolalloc link needed).
INCLUDEPATH += $${_PRO_FILE_PWD_}/../kamepoolalloc

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
    atomic_queue.h \
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
