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
#    $${_PRO_FILE_PWD_}/../../kame/thermometer\
#    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    modbusrtuinterface.h \
    usermotor.h

SOURCES += \
    modbusrtuinterface.cpp \
    usermotor.cpp

#FORMS +=

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../charinterface/release/ -lcharinterface
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../charinterface/debug/ -lcharinterface
else:unix: LIBS += -L$$OUT_PWD/../charinterface/ -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/core/release/ -lmotorcore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/core/debug/ -lmotorcore
else:unix: LIBS += -L$$OUT_PWD/core/ -lmotorcore

INCLUDEPATH += $$PWD/core
DEPENDPATH += $$PWD/core
