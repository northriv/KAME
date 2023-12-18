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

win32:LIBS += -ldmmcore

INCLUDEPATH += $$PWD/../dmm/core
DEPENDPATH += $$PWD/../dmm/core

win32:LIBS += -ldcsourcecore

INCLUDEPATH += $$PWD/../dcsource/core
DEPENDPATH += $$PWD/../dcsource/core
