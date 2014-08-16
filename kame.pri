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

VERSTR = 4.0
DEFINES += VERSION=\'\"$${VERSTR}\"\'

KAME_COREMODULES = coremodules
DEFINES += KAME_COREMODULE_DIR_SURFIX=\'\"/$${KAME_COREMODULES}/\"\'
KAME_COREMODULES2 = coremodules2
DEFINES += KAME_COREMODULE2_DIR_SURFIX=\'\"/$${KAME_COREMODULES2}/\"\'
KAME_MODULES = modules
DEFINES += KAME_MODULE_DIR_SURFIX=\'\"/$${KAME_MODULES}/\"\'

greaterThan(QT_MAJOR_VERSION, 4) {
}
else {
    DEFINES += DATA_INSTALL_DIR=\'\"/usr/share/kame\"\'
}

macx {
    INCLUDEPATH += /opt/local/include
    LIBS += -L/opt/local/lib/ #MacPorts
}

win32 {
#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../boost
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3
    INCLUDEPATH += "C:/Program Files/GnuWin32/include"
    INCLUDEPATH += "C:/Program Files (x86)/GnuWin32/include"
    LIBS += -L"C:/Program Files/GnuWin32/lib/"
    LIBS += -L"C:/Program Files (x86)/GnuWin32/lib/"
    LIBS += -lgsl -lgslcblas -lltdl -lz
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3 -lfftw3-3
}

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += fftw3
unix: PKGCONFIG += gsl
unix: PKGCONFIG += zlib
unix: LIBS += -lltdl

macx: DEFINES += HAVE_LAPACK
