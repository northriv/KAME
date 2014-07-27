PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    fujikininterface.h \
    userflowcontroller.h

SOURCES += \
    fujikininterface.cpp \
    userflowcontroller.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -lflowcontrollercore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
