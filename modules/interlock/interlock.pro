PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

HEADERS += \
    scalarinterlock.h

SOURCES += \
    scalarinterlock.cpp

FORMS += \
    scalarinterlockform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core
