TARGET = transaction_negotiation_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    transaction_negotiation_test.cpp \
    support_standalone.cpp
