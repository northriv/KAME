TARGET = transaction_payload_integrity_3level_mixed_test

include(tests.pri)

QT -= gui core
CONFIG -= qt

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    transaction_payload_integrity_3level_mixed_test.cpp \
    support_standalone.cpp
