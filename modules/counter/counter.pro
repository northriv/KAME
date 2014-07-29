PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    counter.h

SOURCES += \
    counter.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface
