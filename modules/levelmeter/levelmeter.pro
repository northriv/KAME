PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    userlevelmeter.h

SOURCES += \
    userlevelmeter.cpp

#FORMS +=

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

LIBS += -llevelmetercore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
