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
# Depends on kamepoolalloc for the lock-free pool allocator + barrier
# primitives.  Consumers link both:
#   LIBS += -lkamestm -lkamepoolalloc
#
# Production kame.app (kame/kame.pro) inline-compiles these same .cpp
# files (no DLL boundary).  The standalone kamestm dylib exists for
# tests and for downstream consumers that want to embed kamestm
# without the rest of KAME.

CONFIG += plugin
CONFIG -= qt
CONFIG -= app_bundle

CONFIG += c++17
QMAKE_CXXFLAGS += -Wno-register

contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

INCLUDEPATH += $${_PRO_FILE_PWD_}
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
    atomic.h \
    atomic_queue.h \
    atomic_smart_ptr.h \
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

# Link against the sibling kamepoolalloc dylib (built by
# ../kamepoolalloc/kamepoolalloc.pro into a parallel OUT_PWD subdir).
LIBS += -L$$OUT_PWD/../kamepoolalloc -lkamepoolalloc

macx {
    QMAKE_LFLAGS += -Wl,-rpath,@executable_path/../kamepoolalloc
}
unix:!macx {
    QMAKE_LFLAGS += -Wl,-rpath,\\$\$ORIGIN/../kamepoolalloc
}

DESTDIR = $$OUT_PWD
