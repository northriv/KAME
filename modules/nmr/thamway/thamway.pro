PRI_DIR = ../../
include($${PRI_DIR}/modules.pri)

QT += opengl

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\


HEADERS += \
    thamwayprot.h \
    thamwaypulser.h\

SOURCES += \
    thamwayprot.cpp \
    thamwaypulser.cpp \

win32: { #exists("c:\cypress\usb\drivers\ezusbdrv\ezusbsys.h") {
    HEADERS += \
        fx2fw.h\
        cusb.h\
        ezusbthamway.h\
        thamwaydso.h

    SOURCES += \
        cusb.c\
        ezusbthamway.cpp\
        thamwaydso.cpp

    DEFINES += USE_EZUSB
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
