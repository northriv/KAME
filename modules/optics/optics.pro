PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    odmrfm.h \
    useropticalspectrum.h

SOURCES += \
    odmrfm.cpp \
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

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../sg/core
DEPENDPATH += $$PWD/../sg/core

win32:LIBS += -llia

INCLUDEPATH += $$PWD/../lia
DEPENDPATH += $$PWD/../lia

FORMS += \
    core/odmrfmform.ui
