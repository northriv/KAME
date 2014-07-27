PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    usernetworkanalyzer.h

SOURCES += \
    usernetworkanalyzer.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -lnetworkanalyzercore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
