PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

# The Scalar Interlock acts on other drivers through their nodes by name
# (StopMotor, Enabled, RFON, Output, CloseValve, ChannelN, Dump), so it links
# against no driver module.  The Image Trigger reads camera frames
# (XDigitalCamera, in opticscore) and names its files as graph dumps do
# (XGraphNToolBox, in libkame's graph directory).

HEADERS += \
    scalarinterlock.h \
    imagetrigger.h

SOURCES += \
    scalarinterlock.cpp \
    imagetrigger.cpp

FORMS += \
    scalarinterlockform.ui \
    imagetriggerform.ui

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph

win32:LIBS += -lopticscore

INCLUDEPATH += $$PWD/../optics/core
DEPENDPATH += $$PWD/../optics/core
