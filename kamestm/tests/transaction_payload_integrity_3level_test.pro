TARGET = transaction_payload_integrity_3level_test

exists(../../tests/tests.pri) {
    include(../../tests/tests.pri)   # monorepo (kame) — full harness, links libkamepoolalloc + libkamestm
} else {
    include(tests.pri)                 # standalone kamestm repo — Qt-free, no kamepoolalloc, system alloc
}

HEADERS += \
    ../atomic_smart_ptr.h \
    ../transaction.h \
    ../transaction_impl.h \
    ../transaction_definitions.h \
    ../transaction_signal.h \
    ../transaction_negotiation.h \
    ../transaction_neg_impl.h

SOURCES += \
    transaction_payload_integrity_3level_test.cpp
