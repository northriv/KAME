TARGET = transaction_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h\
    ../kame/xtime.h

SOURCES += \
    transaction_test.cpp \
    support.cpp \
    xtime.cpp
