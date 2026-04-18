TARGET = transaction_dynamic_node_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    transaction_dynamic_node_test.cpp \
    support_standalone.cpp
