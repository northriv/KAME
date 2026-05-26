TARGET = atomic_queue_test

include(tests.pri)

HEADERS += \
    ../kame/atomic_queue.h \
    ../kame/atomic_smart_ptr.h \
    ../kame/xthread.h

SOURCES += \
    atomic_queue_test.cpp
