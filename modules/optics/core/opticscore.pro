PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    opticalspectrometer.h

SOURCES += \
    opticalspectrometer.cpp

FORMS += \
    opticalspectrometerform.ui
