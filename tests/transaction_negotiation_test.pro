TARGET = transaction_negotiation_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h\
    ../kame/xtime.h

SOURCES += \
    transaction_negotiation_test.cpp \
    support.cpp \
    xtime.cpp
