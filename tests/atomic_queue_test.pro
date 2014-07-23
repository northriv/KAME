TARGET = atomic_queue_test

TEMPLATE = app

CONFIG += rtti c++11
CONFIG += console
CONFIG += testcase
CONFIG -= app_bundle #macosx

INCLUDEPATH += $${_PRO_FILE_PWD_}/../kame

HEADERS += \
    support.h \
    ../kame/allocator.h

SOURCES += \
    atomic_queue_test.cpp \
    support.cpp
