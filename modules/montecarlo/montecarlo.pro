PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    kamemontecarlo.h \
    montecarlo.h

SOURCES += \
    interaction.cpp \
    kamemontecarlo.cpp \
    montecarlo.cpp

FORMS += \
    montecarloform.ui
