CONFIG += qt exceptions
CONFIG += sse sse2 rtti

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

greaterThan(QT_MAJOR_VERSION, 4) {
	CONFIG += c++11
	DEFINES += QT_NO_OPENGL_ES_2
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}

VERSTR = 4.0
DEFINES += VERSION=\"quotedefined($${VERSTR})\"

KAME_COREMODULES = coremodules
win32-msvc*:DEFINES += KAME_COREMODULE_DIR_SURFIX=\"quotedefined(\\\\$${KAME_COREMODULES}\\\\)\"
else:DEFINES += KAME_COREMODULE_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES}/)\"

KAME_COREMODULES2 = coremodules2
win32-msvc*:DEFINES += KAME_COREMODULE2_DIR_SURFIX=\"quotedefined(\\\\$${KAME_COREMODULES2}\\\\)\"
else:DEFINES += KAME_COREMODULE2_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES2}/)\"

KAME_MODULES = modules
win32-msvc*:DEFINES += KAME_MODULE_DIR_SURFIX=\"quotedefined(\\\\$${KAME_MODULES}\\\\)\"
else:DEFINES += KAME_MODULE_DIR_SURFIX=\"quotedefined(/$${KAME_MODULES}/)\"

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
#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../boost
    DEFINES += GSL_DLL
}
win32-mingw* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/.libs
    LIBS += -lgsl #-lgslcblas
}
win32-msvc* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/
    LIBS += -llibgsl
}

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += gsl

macx: DEFINES += HAVE_LAPACK

#DEFINES += USE_STD_ATOMIC


win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
    QMAKE_LFLAGS += /opt:noref
}
else {
    QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
}
