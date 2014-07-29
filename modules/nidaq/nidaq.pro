PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += opengl

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    nidaqdso.h \
    nidaqmxdriver.h \
    pulserdrivernidaq.h \
    pulserdrivernidaqmx.h

SOURCES += \
    nidaqdso.cpp \
    nidaqmxdriver.cpp \
    pulserdrivernidaq.cpp \
    pulserdrivernidaqmx.cpp

win32:LIBS += -lnmrpulsercore

INCLUDEPATH += $$PWD/../nmr/pulsercore
DEPENDPATH += $$PWD/../nmr/pulsercore

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core
