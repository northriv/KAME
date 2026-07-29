PRI_DIR = ../../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    thamwayprot.h \
    thamwaypulser.h

SOURCES += \
    thamwayprot.cpp \
    thamwaypulser.cpp

FORMS += \
    thamwayprotform.ui
win32: {
    HEADERS += \
        fx2fw.h\
        thamwayusbinterface.h \
        thamwayrealtimedso.h
        thamwaydso.h \

    SOURCES += \
        thamwaydso.cpp \
        thamwayusbinterface.cpp \
        thamwayrealtimedso.cpp

    DEFINES += USE_THAMWAY_USB
}

unix {
    # The probe used to be the MacPorts absolute path only, inside a plain
    # `unix { }` block — so on Linux the whole Thamway Cypress FX2/FX3 USB
    # family (pulser, DSO, realtime DSO) was dropped even with libusb
    # installed.  Same pattern as charinterface.pro.
    macx:exists("/opt/local/include/libusb-1.0/libusb.h"): HAS_LIBUSB = 1
    !macx:system(pkg-config --exists libusb-1.0): HAS_LIBUSB = 1
    !isEmpty(HAS_LIBUSB) {
        macx: LIBS += -lusb-1.0
        else: PKGCONFIG += libusb-1.0
        HEADERS += \
            fx2fw.h\
            thamwayusbinterface.h \
            thamwayrealtimedso.h
            thamwaydso.h \

        SOURCES += \
            thamwaydso.cpp \
            thamwayusbinterface.cpp \
            thamwayrealtimedso.cpp

        DEFINES += USE_THAMWAY_USB
    }
    else {
        message("Missing library for libusb-1.0")
    }
}

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../../charinterface
DEPENDPATH += $$PWD/../../charinterface

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../../sg/core
DEPENDPATH += $$PWD/../../sg/core

win32:LIBS += -lnetworkanalyzercore

INCLUDEPATH += $$PWD/../../networkanalyzer/core
DEPENDPATH += $$PWD/../../networkanalyzer/core

win32:LIBS += -lnmrpulsercore

INCLUDEPATH += $$PWD/../pulsercore
DEPENDPATH += $$PWD/../pulsercore

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../../dso/core
DEPENDPATH += $$PWD/../../dso/core
