PRI_DIR = ../../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    thamwayprot.h \

SOURCES += \
    thamwayprot.cpp \

win32 {
    QT += opengl

    INCLUDEPATH += \
        $${_PRO_FILE_PWD_}/../../kame/graph\

    HEADERS += \
        fx2fw.h\
        cusb.h\
        thamwaypulser.h

    SOURCES += \
        cusb.c\
        thamwaypulser.cpp

    win32:LIBS += -lnmrpulsercore

    INCLUDEPATH += $$PWD/../pulsercore
    DEPENDPATH += $$PWD/../pulsercore
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
