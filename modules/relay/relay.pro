PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

INCLUDEPATH += \
    $${OUT_PWD}/core\ #for ui_*.h

HEADERS += \
    userrelay.h

SOURCES += \
    userrelay.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lrelaycore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core
