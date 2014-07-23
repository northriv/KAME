TARGET = transaction_negotiation_test

TEMPLATE = app

CONFIG += rtti c++11
CONFIG += console
CONFIG += testcase
CONFIG -= app_bundle #macosx

INCLUDEPATH += $${_PRO_FILE_PWD_}/../kame

HEADERS += \
    support.h \
    ../kame/allocator.h\
    ../kame/xtime.h

SOURCES += \
    transaction_negotiation_test.cpp \
    support.cpp \
    xtime.cpp
