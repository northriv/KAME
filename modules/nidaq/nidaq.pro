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

unix:!macx {
    # NI-DAQmx ships a Linux edition (ni-daqmx / "NI-DAQmx for Linux"): the
    # ANSI C header lands in /usr/include and libnidaqmx.so in the default
    # library path.  Without this probe HAVE_NI_DAQMX was never defined
    # outside win32, and the four drivers here were compiled against stubs.
    # (macOS has no NI-DAQmx at all — NI dropped it after 10.13 — so there is
    # deliberately no macx branch.)
    NIDAQMX_H = /usr/include/NIDAQmx.h
    !exists($${NIDAQMX_H}): NIDAQMX_H = /usr/local/include/NIDAQmx.h
    !exists($${NIDAQMX_H}): NIDAQMX_H = /usr/local/natinst/nidaqmx/include/NIDAQmx.h
    exists($${NIDAQMX_H}) {
        INCLUDEPATH += $$dirname(NIDAQMX_H)
        LIBS += -lnidaqmx
        DEFINES += HAVE_NI_DAQMX
        message("using NI-DAQmx from $${NIDAQMX_H}.")
    }
    else {
        message("NI-DAQmx not found: the NIDAQmxDSO and NI-DAQ pulser drivers \
will NOT be registered (install ni-daqmx to enable them).")
    }
}

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
