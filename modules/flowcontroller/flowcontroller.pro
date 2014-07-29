PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    fujikininterface.h \
    userflowcontroller.h

SOURCES += \
    fujikininterface.cpp \
    userflowcontroller.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lflowcontrollercore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
