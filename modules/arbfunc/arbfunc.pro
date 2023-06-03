PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \

HEADERS += \
    arbfunc.h \
    userarbfunc.h

SOURCES += \
    arbfunc.cpp \
    userarbfunc.cpp

FORMS += \
    arbfuncform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface
