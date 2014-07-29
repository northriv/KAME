PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    usernetworkanalyzer.h

SOURCES += \
    usernetworkanalyzer.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lnetworkanalyzercore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
