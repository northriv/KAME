PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    pulserdriver.h\
    pulserdriverconnector.h \
    softtrigger.h

SOURCES += \
    pulserdriver.cpp\
    pulserdriverconnector.cpp \
    softtrigger.cpp

FORMS += \
    pulserdriverform.ui\
    pulserdrivermoreform.ui
