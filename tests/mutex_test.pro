TARGET = mutex_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    mutex_test.cpp \
    support_standalone.cpp
