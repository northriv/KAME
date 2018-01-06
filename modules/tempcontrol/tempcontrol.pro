PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/thermometer\

HEADERS += \
    tempcontrol.h\
    usertempcontrol.h \
    omronmodbus.h \
    tempmanager.h

SOURCES += \
    tempcontrol.cpp\
    usertempcontrol.cpp \
    omronmodbus.cpp \
    tempmanager.cpp

FORMS +=\
    tempcontrolform.ui \
    tempmanagerform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -ldcsourcecore

INCLUDEPATH += $$PWD/../dcsource/core
DEPENDPATH += $$PWD/../dcsource/core

win32:LIBS += -lflowcontrollercore

INCLUDEPATH += $$PWD/../flowcontroller/core
DEPENDPATH += $$PWD/../flowcontroller/core
