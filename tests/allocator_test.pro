TARGET = allocator_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h

SOURCES += \
    allocator_test.cpp \
    support.cpp
