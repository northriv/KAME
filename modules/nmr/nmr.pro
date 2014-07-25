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
    $${_PRO_FILE_PWD_}/../../kame/icons\

INCLUDEPATH += $$OUT_PWD/../../kame
DEPENDPATH += $$OUT_PWD/../../kame

HEADERS += \
    autolctuner.h \
    nmrfspectrum.h \
    nmrpulse.h \
    nmrrelax.h \
    nmrrelaxfit.h \
    nmrspectrum.h \
    nmrspectrumbase_impl.h \
    nmrspectrumbase.h \
    nmrspectrumsolver.h \
    pulseanalyzer.h \

SOURCES += \
    autolctuner.cpp \
    nmrfspectrum.cpp \
    nmrpulse.cpp \
    nmrrelax.cpp \
    nmrrelaxfit.cpp \
    nmrspectrum.cpp \
    nmrspectrumsolver.cpp \
    pulseanalyzer.cpp \

FORMS += \
    autolctunerform.ui \
    nmrfspectrumform.ui \
    nmrpulseform.ui \
    nmrrelaxform.ui \
    nmrspectrumform.ui

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

macx {
    INCLUDEPATH += /opt/local/include
    DEPENDPATH += /opt/local/include
}

unix: PKGCONFIG += fftw3
unix: PKGCONFIG += gsl

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

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../sg/core/release/ -lsgcore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../sg/core/debug/ -lsgcore
else:unix: LIBS += -L$$OUT_PWD/../sg/core/ -lsgcore

INCLUDEPATH += $$PWD/../sg/core
DEPENDPATH += $$PWD/../sg/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../dso/core/release/ -ldsocore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../dso/core/debug/ -ldsocore
else:unix: LIBS += -L$$OUT_PWD/../dso/core/ -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../motor/core/release/ -lmotorcore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../motor/core/debug/ -lmotorcore
else:unix: LIBS += -L$$OUT_PWD/../motor/core/ -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../networkanalyzer/core/release/ -lnetworkanalyzercore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../networkanalyzer/core/debug/ -lnetworkanalyzercore
else:unix: LIBS += -L$$OUT_PWD/../networkanalyzer/core/ -lnetworkanalyzercore

INCLUDEPATH += $$PWD/../networkanalyzer/core
DEPENDPATH += $$PWD/../networkanalyzer/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../dmm/core/release/ -ldmmcore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../dmm/core/debug/ -ldmmcore
else:unix: LIBS += -L$$OUT_PWD/../dmm/core/ -ldmmcore

INCLUDEPATH += $$PWD/../dmm/core
DEPENDPATH += $$PWD/../dmm/core

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../magnetps/core/release/ -lmagnetpscore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../magnetps/core/debug/ -lmagnetpscore
else:unix: LIBS += -L$$OUT_PWD/../magnetps/core/ -lmagnetpscore

INCLUDEPATH += $$PWD/../magnetps/core
DEPENDPATH += $$PWD/../magnetps/core
