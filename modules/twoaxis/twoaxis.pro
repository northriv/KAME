PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    twoaxis.h

SOURCES += \
    twoaxis.cpp

FORMS += \
    twoaxisform.ui

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core
