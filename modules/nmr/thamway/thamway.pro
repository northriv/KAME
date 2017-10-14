PRI_DIR = ../../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    thamwayprot.h \
    thamwaypulser.h \
    fx2fw.h\
    cyfxusb.h\
    thamwaydso.h \
    thamwayusbinterface.h \
    cyfxusbinterface_impl.h \
    thamwayrealtimedso.h

SOURCES += \
    thamwayprot.cpp \
    thamwaypulser.cpp \
    thamwaydso.cpp \
    cyfxusb.cpp\
    thamwayusbinterface.cpp \
    thamwayrealtimedso.cpp

FORMS += \
    thamwayprotform.ui
win32: {
    HEADERS += \
        cyfxusb_win32.h\

    SOURCES += \
        cyfxusb_win32.cpp\

    DEFINES += USE_THAMWAY_USB

    LIBS += -lsetupapi
}

unix {
    exists("/opt/local/include/libusb-1.0/libusb.h") {
        LIBS += -lusb-1.0
        HEADERS += \

        SOURCES += \
            cyfxusb_libusb.cpp\

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
