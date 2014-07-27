PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += $$OUT_PWD/../../kame
DEPENDPATH += $$OUT_PWD/../../kame

HEADERS += \
    pulserdriverh8.h \
    pulserdriversh.h

SOURCES += \
    pulserdriverh8.cpp \
    pulserdriversh.cpp

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -lnmrpulsercore

INCLUDEPATH += $$PWD/pulsercore
DEPENDPATH += $$PWD/pulsercore

