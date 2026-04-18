TARGET = transaction_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    transaction_test.cpp \
    support_standalone.cpp
