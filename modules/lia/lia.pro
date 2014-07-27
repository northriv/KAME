PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    lockinamp.h \
    userlockinamp.h

SOURCES += \
    lockinamp.cpp \
    userlockinamp.cpp

FORMS += \
    lockinampform.ui

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

