PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    useropticalspectrum.h

SOURCES += \
    useropticalspectrum.cpp

unix {
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
}

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lopticscore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
