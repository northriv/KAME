PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    modbusrtuinterface.h \
    usermotor.h

SOURCES += \
    modbusrtuinterface.cpp \
    usermotor.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -lmotorcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
