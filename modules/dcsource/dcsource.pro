PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    userdcsource.h

SOURCES += \
    userdcsource.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS +=-ldcsourcecore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
