PRI_DIR = ../../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    thamwayprot.h \
    thamwaypulser.h \

SOURCES += \
    thamwayprot.cpp \
    thamwaypulser.cpp

FORMS += \
    thamwayprotform.ui
win32: {
    exists("c:\cypress\lib\CyAPI.lib") {
        HEADERS += \
            cusb2.h

        SOURCES += \
            cusb2.c\

    }
#exists("c:\cypress\usb\drivers\ezusbdrv\ezusbsys.h") {
    HEADERS += \
        fx2fw.h\
        cusb.h\
        ezusbthamway.h\
        cyfxusb.h\
        thamwayfxusb.h\
        thamwaydso.h

    SOURCES += \
        cusb.c\
        ezusbthamway.cpp\
        cyfxusb.cpp\
        thamwayfxusb.cpp\
        thamwaydso.cpp

    DEFINES += USE_THAMWAY_USB
    DEFINES += USE_THAMWAY_USB_FX2FW
}

unix {
    exists("/opt/local/include/libusb-1.0/libusb.h") {
        LIBS += -lusb-1.0
        HEADERS += \
            libusb2cusb.h\
            ezusbthamway.h\
            thamwaydso.h

        SOURCES += \
            libusb2cusb.cpp\
            ezusbthamway.cpp\
            thamwaydso.cpp

        DEFINES += USE_THAMWAY_USB
        DEFINES += USE_THAMWAY_USB_LIBUSB
        macx: DEFINES += KAME_THAMWAY_USB_DIR=\"quotedefined(Contents/Resources/)\"
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
