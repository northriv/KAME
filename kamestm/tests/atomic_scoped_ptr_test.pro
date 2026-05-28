TARGET = atomic_scoped_ptr_test

include(../../tests/tests.pri)

HEADERS += \
    ../kame/atomic_smart_ptr.h \
    ../kame/xthread.h

SOURCES += \
    atomic_scoped_ptr_test.cpp
