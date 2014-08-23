TARGET = transaction_dynamic_node_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h\
    ../kame/xtime.h

SOURCES += \
    transaction_dynamic_node_test.cpp \
    support.cpp \
    xtime.cpp
