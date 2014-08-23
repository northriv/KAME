TARGET = mutex_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h

SOURCES += \
    mutex_test.cpp \
    support.cpp
