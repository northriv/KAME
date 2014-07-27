PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    usermagnetps.h

SOURCES += \
    usermagnetps.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -lmagnetpscore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
