PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    modbusrtuinterface.h \
    usermotor.h

SOURCES += \
    modbusrtuinterface.cpp \
    usermotor.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
