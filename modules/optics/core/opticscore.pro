PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    lasermodule.h \
    opticalspectrometer.h

SOURCES += \
    lasermodule.cpp \
    opticalspectrometer.cpp

FORMS += \
    lasermoduleform.ui \
    opticalspectrometerform.ui
