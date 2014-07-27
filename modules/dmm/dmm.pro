PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    userdmm.h

SOURCES += \
    userdmm.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -ldmmcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
