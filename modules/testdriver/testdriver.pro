TEMPLATE = lib

CONFIG += plugin

PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    testdriver.h

SOURCES += \
    testdriver.cpp

