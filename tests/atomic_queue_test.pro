TARGET = atomic_queue_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    atomic_queue_test.cpp \
    support_standalone.cpp
