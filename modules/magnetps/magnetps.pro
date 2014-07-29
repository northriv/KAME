PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    usermagnetps.h

SOURCES += \
    usermagnetps.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lmagnetpscore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
