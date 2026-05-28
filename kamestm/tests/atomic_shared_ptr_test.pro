TARGET = atomic_shared_ptr_test

include(../../tests/tests.pri)

HEADERS += \
    ../kame/atomic_smart_ptr.h \
    ../kame/xthread.h

SOURCES += \
    atomic_shared_ptr_test.cpp
