CONFIG += qt exceptions
CONFIG += rtti
contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

QT       += core gui

#remove these two to use QOpenGLWidget
#DEFINES += USE_QGLWIDGET
#QT		 += opengl

greaterThan(QT_MAJOR_VERSION, 5): QT += opengl openglwidgets
#For QTextCodec
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

greaterThan(QT_MAJOR_VERSION, 3) {
	CONFIG += c++11
	#For ruby.h
	QMAKE_CXXFLAGS += -Wno-register
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}

VERSTR = 5.5
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

DEFINES += EIGEN_NO_DEBUG

macx {
    INCLUDEPATH += /opt/local/include
    INCLUDEPATH += /opt/local/include/eigen3
    LIBS += -L/opt/local/lib/ #MacPorts
}

win32 {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../eigen3

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

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
    QMAKE_LFLAGS += /opt:noref
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
win32-g++ {
#workaround for movaps alignment problem
    QMAKE_CXXFLAGS += -mstackrealign
#increases stack size to 8MB, the same as Linux/OS X.
    QMAKE_CXXFLAGS += -Wl,--stack,8388608
}
