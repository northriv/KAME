PRI_DIR = ../../
include($${PRI_DIR}/modules-shared.pri)

QT += opengl

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../../kame/graph\

HEADERS += \
    pulserdriver.h\
    pulserdriverconnector.h

SOURCES += \
    pulserdriver.cpp\
    pulserdriverconnector.cpp

FORMS += \
    pulserdriverform.ui\
    pulserdrivermoreform.ui
