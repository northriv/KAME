# Microbenchmark for the Snapshot/Transaction lookup memo — build and run
# manually; intentionally NOT registered in tests/tests.pro (perf tool,
# not a correctness testcase).
TARGET = transaction_lookup_bench

exists(../../tests/tests.pri) {
    include(../../tests/tests.pri)   # monorepo (kame) — full harness, links libkamepoolalloc + libkamestm
} else {
    include(tests.pri)                 # standalone kamestm repo — Qt-free, no kamepoolalloc, system alloc
}
CONFIG -= testcase

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
    transaction_lookup_bench.cpp
