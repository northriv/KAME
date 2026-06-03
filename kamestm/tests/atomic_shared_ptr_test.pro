TARGET = atomic_shared_ptr_test

exists(../../tests/tests.pri) {
    include(../../tests/tests.pri)   # monorepo (kame) — full harness, links libkamepoolalloc + libkamestm
} else {
    include(tests.pri)                 # standalone kamestm repo — Qt-free, no kamepoolalloc, system alloc
}

HEADERS += \
    ../atomic_smart_ptr.h \
    ../xthread.h

SOURCES += \
    atomic_shared_ptr_test.cpp
