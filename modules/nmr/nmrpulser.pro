TEMPLATE = lib

CONFIG += plugin
CONFIG += qt exceptions
CONFIG += sse2 rtti

QT       += core gui opengl
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
#    $${_PRO_FILE_PWD_}/../../kame/thermometer\
    $${_PRO_FILE_PWD_}/../../kame/math\

INCLUDEPATH += $$OUT_PWD/../../kame
DEPENDPATH += $$OUT_PWD/../../kame

HEADERS += \
    pulserdriverh8.h \
    pulserdriversh.h

SOURCES += \
    pulserdriverh8.cpp \
    pulserdriversh.cpp

FORMS += \

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

macx {
    INCLUDEPATH += /opt/local/include
    DEPENDPATH += /opt/local/include
}

unix: PKGCONFIG += fftw3

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../charinterface/release/ -lcharinterface
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../charinterface/debug/ -lcharinterface
else:unix: LIBS += -L$$OUT_PWD/../charinterface/ -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/pulsercore/release/ -lnmrpulsercore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/pulsercore/debug/ -lnmrpulsercore
else:unix: LIBS += -L$$OUT_PWD/pulsercore/ -lnmrpulsercore

INCLUDEPATH += $$PWD/pulsercore
DEPENDPATH += $$PWD/pulsercore

