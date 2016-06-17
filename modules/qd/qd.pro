PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    userqdppms.h \

SOURCES += \
    userqdppms.cpp \

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lqdcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core

