TARGET = atomic_queue_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h

SOURCES += \
    atomic_queue_test.cpp \
    support.cpp
