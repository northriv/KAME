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

win32 {
#    INCLUDEPATH +="C:\Program Files (x86)\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include"
    INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
    LIBS += -lnidaqmx
    DEFINES += HAVE_NI_DAQMX
}
