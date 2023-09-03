PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    digitalcamera.h \
    lasermodule.h \
    opticalspectrometer.h

SOURCES += \
    digitalcamera.cpp \
    lasermodule.cpp \
    opticalspectrometer.cpp

FORMS += \
    digitalcameraform.ui \
    lasermoduleform.ui \
    opticalspectrometerform.ui
