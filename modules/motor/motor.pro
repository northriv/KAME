PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    cam.h \
    usermotor.h

SOURCES += \
    cam.cpp \
    usermotor.cpp

FORMS += \
    camform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
