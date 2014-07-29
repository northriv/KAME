PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    userdmm.h

SOURCES += \
    userdmm.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -ldmmcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
