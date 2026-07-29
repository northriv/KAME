PRI_DIR = ../../
# Depends on another core module, so it must load after coremodules/.
KAME_MODULE_TIER = coremodules2
include($${PRI_DIR}/modules-shared.pri)

win32:LIBS += -ldcsourcecore

INCLUDEPATH += $$PWD/../../dcsource/core
DEPENDPATH += $$PWD/../../dcsource/core

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    filterwheel.h \
    digitalcamera.h \
    imageprocessor.h \
    lasermodule.h \
    opticalspectrometer.h\
    spectralmathtool.h

SOURCES += \
    filterwheel.cpp \
    digitalcamera.cpp \
    imageprocessor.cpp \
    lasermodule.cpp \
    opticalspectrometer.cpp\
    spectralmathtool.cpp

FORMS += \
    digitalcameraform.ui \
    filterwheelform.ui \
    imageprocessorform.ui \
    lasermoduleform.ui \
    opticalspectrometerform.ui
