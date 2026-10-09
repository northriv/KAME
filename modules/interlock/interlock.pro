PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

# Acts on other drivers through their nodes by name (StopMotor, Enabled,
# RFON, Output, CloseValve, ChannelN), so it links against no driver module.

HEADERS += \
    scalarinterlock.h

SOURCES += \
    scalarinterlock.cpp

FORMS += \
    scalarinterlockform.ui
