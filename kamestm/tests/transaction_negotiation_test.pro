TARGET = transaction_negotiation_test

exists(../../tests/tests.pri) {
    include(../../tests/tests.pri)   # monorepo (kame) — full harness, links libkamepoolalloc + libkamestm
} else {
    include(tests.pri)                 # standalone kamestm repo — Qt-free, no kamepoolalloc, system alloc
}

HEADERS += \
    ../atomic_smart_ptr.h \
    ../xthread.h \
    ../transaction.h \
    ../transaction_impl.h \
    ../transaction_definitions.h \
    ../transaction_signal.h \
    ../transaction_negotiation.h \
    ../transaction_neg_impl.h

SOURCES += \
    transaction_negotiation_test.cpp
