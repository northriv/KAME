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
    $${_PRO_FILE_PWD_}/../../kame/math\
#    $${_PRO_FILE_PWD_}/../../kame/thermometer\
    $${_PRO_FILE_PWD_}/../../kame/graph\

HEADERS += \
    nidaqdso.h \
    nidaqmxdriver.h \
    pulserdrivernidaq.h \
    pulserdrivernidaqmx.h

SOURCES += \
    nidaqdso.cpp \
    nidaqmxdriver.cpp \
    pulserdrivernidaq.cpp \
    pulserdrivernidaqmx.cpp

#FORMS +=

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}
macx {
    INCLUDEPATH += /opt/local/include
    DEPENDPATH += /opt/local/include
}

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../nmr/pulsercore/release/ -lnmrpulsercore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../nmr/pulsercore/debug/ -lnmrpulsercore
else:unix: LIBS += -L$$OUT_PWD/../nmr/pulsercore/ -lnmrpulsercore

INCLUDEPATH += $$PWD/../nmr/pulsercore
DEPENDPATH += $$PWD/../nmr/pulsercore

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../dso/core/release/ -ldsocore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../dso/core/debug/ -ldsocore
else:unix: LIBS += -L$$OUT_PWD/../dso/core/ -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core
