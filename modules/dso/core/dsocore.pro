PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    dso.h \
    dsorealtimeacq.h

SOURCES += \
    dso.cpp \
    dsorealtimeacq.cpp

FORMS += \
    dsoform.ui

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../../sg/core
DEPENDPATH += $$PWD/../../sg/core

win32 {
    DESTDIR=$$OUT_PWD/../../../coremodules2
}
