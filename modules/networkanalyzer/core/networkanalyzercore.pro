PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += opengl

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    networkanalyzer.h

SOURCES += \
    networkanalyzer.cpp

FORMS += \
    networkanalyzerform.ui
