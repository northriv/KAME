TEMPLATE = lib

CONFIG += plugin
CONFIG += qt exceptions
CONFIG += sse2 rtti

QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

greaterThan(QT_MAJOR_VERSION, 4) {
	CONFIG += c++11
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}

VERSTR = '\\"4.0\\"'
DEFINES += VERSION=\"$${VERSTR}\"
DEFINES += KAME_MODULE_DIR_SURFIX=\'\"/kame/modules\"\'
greaterThan(QT_MAJOR_VERSION, 4) {
}
else {
    DEFINES += DATA_INSTALL_DIR=\'\"/usr/share/kame\"\'
}

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame\
    $${_PRO_FILE_PWD_}/../../kame/analyzer\
    $${_PRO_FILE_PWD_}/../../kame/driver\
    $${_PRO_FILE_PWD_}/../../kame/math\
    $${_PRO_FILE_PWD_}/../../kame/thermometer\
#    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    tempcontrol.h\
    usertempcontrol.h

SOURCES += \
    tempcontrol.cpp\
    usertempcontrol.cpp

FORMS +=\
    tempcontrolform.ui

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

macx {
    INCLUDEPATH += /opt/local/include
    DEPENDPATH += /opt/local/include
}

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../charinterface/release/ -lcharinterface
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../charinterface/debug/ -lcharinterface
else:unix: LIBS += -L$$OUT_PWD/../charinterface/ -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../dcsource/core/release/ -ldcsourcecore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../dcsource/core/debug/ -ldcsourcecore
else:unix: LIBS += -L$$OUT_PWD/../dcsource/core/ -ldcsourcecore

INCLUDEPATH += $$PWD/../dcsource/core
DEPENDPATH += $$PWD/../dcsource/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../flowcontroller/core/release/ -lflowcontrollercore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../flowcontroller/core/debug/ -lflowcontrollercore
else:unix: LIBS += -L$$OUT_PWD/../flowcontroller/core/ -lflowcontrollercore

INCLUDEPATH += $$PWD/../flowcontroller/core
DEPENDPATH += $$PWD/../flowcontroller/core
