PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    lecroy.h \
    tds.h \

SOURCES += \
    lecroy.cpp \
    tds.cpp

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
