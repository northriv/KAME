PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += opengl

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    dso.h

SOURCES += \
    dso.cpp

FORMS += \
    dsoform.ui

LIBS += -lsgcore

INCLUDEPATH += $$PWD/../../sg/core
DEPENDPATH += $$PWD/../../sg/core
