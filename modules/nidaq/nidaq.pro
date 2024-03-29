PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

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
    exists(C:/Program Files/National Instruments/NI-DAQ/DAQmx ANSI C Dev/include/NIDAQmx.h) {
        INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include"
        contains(QMAKE_HOST.arch, x86_64) {
            LIBS += -L"C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\lib64\msvc" -lNIDAQmx
        }
        else {
            LIBS += -L"C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\lib32\msvc" -lNIDAQmx
        }
        DEFINES += HAVE_NI_DAQMX
    }

    else {
        exists(C:/Program Files (x86)/National Instruments/NI-DAQ/DAQmx ANSI C Dev/include/NIDAQmx.h) {
            INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
            contains(QMAKE_HOST.arch, x86_64) {
                LIBS += -L"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib64\msvc" -lNIDAQmx
            }
            else {
                LIBS += -L"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib32\msvc" -lNIDAQmx
            }
            DEFINES += HAVE_NI_DAQMX
        }
        else {
            exists(C:/NI-DAQ/DAQmx ANSI C Dev/include/NIDAQmx.h) {
                INCLUDEPATH += "C:\NI-DAQ\DAQmx ANSI C Dev\include"
                LIBS += -L"C:\NI-DAQ\DAQmx ANSI C Dev\lib\msvc" -lNIDAQmx
                DEFINES += HAVE_NI_DAQMX
            }
            else {
                message("Missing library for NI DAQmx")
            }
        }
    }
}
