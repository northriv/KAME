PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    qdppms.h \

SOURCES += \
    qdppms.cpp \

FORMS += \
    qdppmsform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

