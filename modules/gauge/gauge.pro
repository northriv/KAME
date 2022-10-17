PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \

HEADERS += \
    gauge.h \
    usergauge.h

SOURCES += \
    gauge.cpp \
    usergauge.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface
