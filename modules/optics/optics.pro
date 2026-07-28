PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\
    $${OUT_PWD}/core\ #for ui_*.h

HEADERS += \
    filterwheelstmdriven.h \
    odmr2danalysis.h \
    odmrfm.h \
    odmrimaging.h \
    userlasermodule.h \
    useropticalspectrum.h \
    odmrfspectrum.h

SOURCES += \
    filterwheelstmdriven.cpp \
    odmr2danalysis.cpp \
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
    odmr2danalysisform.ui \
    odmrfmform.ui

unix {
    # These three probes were all macOS-absolute paths (MacPorts /opt/local,
    # Homebrew-ish /usr/local/opt) inside a plain `unix { }` block, so on
    # Linux they could never match and the IIDC camera, Ocean Optics
    # spectrometer and Euresys frame-grabber drivers silently disappeared from
    # a build that had every one of those libraries installed.  Keep the
    # MacPorts paths for macOS and use pkg-config elsewhere.
    macx:exists("/opt/local/include/dc1394/dc1394.h"): HAS_DC1394 = 1
    !macx:system(pkg-config --exists libdc1394-2): HAS_DC1394 = 1
    !isEmpty(HAS_DC1394) {
        macx: LIBS += -ldc1394
        else: PKGCONFIG += libdc1394-2
        HEADERS += \
            iidccamera.h \

        SOURCES += \
            iidccamera.cpp \

        DEFINES += USE_LIBDC1394
    }
    else {
        message("Missing library for libdc1394")
    }
    macx:exists("/opt/local/include/libusb-1.0/libusb.h"): HAS_LIBUSB = 1
    !macx:system(pkg-config --exists libusb-1.0): HAS_LIBUSB = 1
    !isEmpty(HAS_LIBUSB) {
        macx: LIBS += -lusb-1.0
        else: PKGCONFIG += libusb-1.0
        HEADERS += \
            oceanopticsusb.h \

        SOURCES += \
            oceanopticsusb.cpp \

        DEFINES += USE_OCEANOPTICS_USB
    }
    else {
        message("Missing library for libusb-1.0")
    }
    # Euresys installs under /opt/euresys/egrabber on Linux and
    # /usr/local/opt/euresys/egrabber on macOS; EGENTL_HOME overrides both.
    EGRABBER_DIR = $$(EGENTL_HOME)
    isEmpty(EGRABBER_DIR): EGRABBER_DIR = /usr/local/opt/euresys/egrabber
    !exists($${EGRABBER_DIR}/include/EGrabber.h): EGRABBER_DIR = /opt/euresys/egrabber
    exists($${EGRABBER_DIR}/include/EGrabber.h) {
        INCLUDEPATH += $${EGRABBER_DIR}/include/
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
