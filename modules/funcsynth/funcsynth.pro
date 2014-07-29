PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

HEADERS += \
    funcsynth.h \
    userfuncsynth.h

SOURCES += \
    funcsynth.cpp \
    userfuncsynth.cpp

FORMS += \
    funcsynthform.ui

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface
