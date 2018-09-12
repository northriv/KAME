PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \

HEADERS += \
    pumpcontroller.h

SOURCES += \
    pumpcontroller.cpp \
    userpumpcontroller.cpp

FORMS += \
    pumpcontrollerform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface
