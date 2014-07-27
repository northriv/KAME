PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    fourres.h

SOURCES += \
    fourres.cpp

FORMS += \
    fourresform.ui

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -ldmmcore

INCLUDEPATH += $$PWD/../dmm/core
DEPENDPATH += $$PWD/../dmm/core

LIBS += -ldcsourcecore

INCLUDEPATH += $$PWD/../dcsource/core
DEPENDPATH += $$PWD/../dcsource/core
