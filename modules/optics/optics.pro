PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\
    $${OUT_PWD}/core\ #for ui_*.h

HEADERS += \
    filterwheelstmdriven.h \
    odmrfm.h \
    odmrimaging.h \
    userlasermodule.h \
    useropticalspectrum.h \
    odmrfspectrum.h

SOURCES += \
    filterwheelstmdriven.cpp \
    odmrfm.cpp \
    odmrimaging.cpp \
    userlasermodule.cpp \
    useropticalspectrum.cpp \
    odmrfspectrum.cpp

exists("odmrimagingng.cpp") {
HEADERS += \
    odmrimagingng.h \

SOURCES += \
    odmrimagingng.cpp \
}

FORMS += \
    odmrimagingform.ui \
    odmrfspectrumform.ui \
    odmrfmform.ui

unix {
    exists("/opt/local/include/dc1394/dc1394.h") {
        LIBS += -ldc1394
        HEADERS += \
            iidccamera.h \

        SOURCES += \
            iidccamera.cpp \

        DEFINES += USE_LIBDC1394
    }
    else {
        message("Missing library for libdc1394")
    }
    exists("/opt/local/include/libusb-1.0/libusb.h") {
        LIBS += -lusb-1.0
        HEADERS += \
            oceanopticsusb.h \

        SOURCES += \
            oceanopticsusb.cpp \

        DEFINES += USE_OCEANOPTICS_USB
    }
    else {
        message("Missing library for libusb-1.0")
    }
    exists("/usr/local/opt/euresys/egrabber/include/EGrabber.h") {
        INCLUDEPATH += /usr/local/opt/euresys/egrabber/include/
        HEADERS += \
            euresyscamera.h

        SOURCES += \
            euresyscamera.cpp \

        DEFINES += USE_EURESYS_EGRABBER
    }
    else {
        message("Missing library for egrabber")
    }
}

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lopticscore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../sg/core
DEPENDPATH += $$PWD/../sg/core

win32:LIBS += -lliacore

INCLUDEPATH += $$PWD/../lia/core
DEPENDPATH += $$PWD/../lia/core

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core
