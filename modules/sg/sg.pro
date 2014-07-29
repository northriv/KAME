PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    usersignalgenerator.h

SOURCES += \
    usersignalgenerator.cpp

#FORMS +=

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
