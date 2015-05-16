CONFIG += qt exceptions
CONFIG += sse sse2 rtti

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

VERSTR = 4.0
DEFINES += VERSION=\"quotedefined($${VERSTR})\"

KAME_COREMODULES = coremodules
DEFINES += KAME_COREMODULE_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES}/)\"

KAME_COREMODULES2 = coremodules2
DEFINES += KAME_COREMODULE2_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES2}/)\"

KAME_MODULES = modules
DEFINES += KAME_MODULE_DIR_SURFIX=\"quotedefined(/$${KAME_MODULES}/)\"

greaterThan(QT_MAJOR_VERSION, 4) {
}
else {
    DEFINES += DATA_INSTALL_DIR=\"\"quotedefined(/usr/share/kame)\"
}

macx {
    INCLUDEPATH += /opt/local/include
    LIBS += -L/opt/local/lib/ #MacPorts
}

win32 {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3

#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../boost
    DEFINES += GSL_DLL
}
win32-g++ {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/.libs
    LIBS += -lgsl #-lgslcblas
    LIBS += -lfftw3-3
}
win32-msvc* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/
    LIBS += -llibgsl
    LIBS += -llibfftw3-3
}

unix {
    CONFIG += link_pkgconfig
    macx {
        LIBS += -lgsl -lgslcblas -lm
    }
    else {
        PKGCONFIG += gsl
    }
}

#macx: DEFINES += HAVE_LAPACK

#DEFINES += USE_STD_ATOMIC


win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
    QMAKE_LFLAGS += /opt:noref
}
else {
    QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
}
