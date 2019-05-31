PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

QT += widgets

INCLUDEPATH += \
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

#    pulseanalyzer.h \

SOURCES += \
    autolctuner.cpp \
    nmrfspectrum.cpp \
    nmrpulse.cpp \
    nmrrelax.cpp \
    nmrrelaxfit.cpp \
    nmrspectrum.cpp \
    nmrspectrumsolver.cpp \

#    pulseanalyzer.cpp \

FORMS += \
    autolctunerform.ui \
    nmrfspectrumform.ui \
    nmrpulseform.ui \
    nmrrelaxform.ui \
    nmrspectrumform.ui

unix {
    macx {
        LIBS += -lfftw3
        LIBS += -lgsl -lgslcblas
    }
    else {
        PKGCONFIG += fftw3
        PKGCONFIG += gsl
    }
#    LIBS += -lclapack -lcblas -latlas
}

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS +=  -lnmrpulsercore

INCLUDEPATH += $$PWD/pulsercore
DEPENDPATH += $$PWD/pulsercore

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../sg/core
DEPENDPATH += $$PWD/../sg/core

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core

win32:LIBS += -lnetworkanalyzercore

INCLUDEPATH += $$PWD/../networkanalyzer/core
DEPENDPATH += $$PWD/../networkanalyzer/core

win32:LIBS += -ldmmcore

INCLUDEPATH += $$PWD/../dmm/core
DEPENDPATH += $$PWD/../dmm/core

win32:LIBS += -lmagnetpscore

INCLUDEPATH += $$PWD/../magnetps/core
DEPENDPATH += $$PWD/../magnetps/core

win32:LIBS += -lqdcore

INCLUDEPATH += $$PWD/../qd/core
DEPENDPATH += $$PWD/../qd/core

