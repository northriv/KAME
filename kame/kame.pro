TARGET = kame
TEMPLATE = app

CONFIG += qt exceptions
CONFIG += sse2 rtti

greaterThan(QT_MAJOR_VERSION, 4) {
	CONFIG += c++11
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}

QT       += core gui opengl
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

VERSTR = '\\"4.0\\"'
DEFINES += VERSION=\"$${VERSTR}\"
KAME_MODULES = modules
DEFINES += KAME_MODULE_DIR_SURFIX=\'\"/$${KAME_MODULES}/\"\'

greaterThan(QT_MAJOR_VERSION, 4) {
}
else {
	DEFINES += DATA_INSTALL_DIR=\'\"/usr/share/kame\"\'
}

SOURCES += main.cpp \


RESOURCES += \
    kame.qrc

TRANSLATIONS = ../po/ja/kame.po.ts

scriptfile.files = script/rubylineshell.rb
macx {
    scriptfile.path = Contents/Resources
    QMAKE_BUNDLE_DATA += scriptfile
}
else {
    unix {
        INSTALLS += scriptfile
    }
    else {
        DISTFILES += script/rubylineshell.rb
    }
}

macx: ICON = kame.icns

macx {
 LIBS += -framework Ruby
}
else {
    win32 {
        LIBS += -L"C:/Ruby187/lib/" -lmsvcrt-ruby18-static
    }
    unix: LIBS += -lruby
}

macx {
    LIBS += -L/opt/local/lib/ #MacPorts
}
win32 {
    LIBS += -L"C:/Program Files (x86)/GnuWin32/lib/"
}

macx {
  QMAKE_LFLAGS += -all_load -dynamic
}

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += fftw3
unix: PKGCONFIG += gsl
unix: PKGCONFIG += zlib
unix: LIBS += -lltdl
unix: LIBS += -lclapack -lcblas -latlas

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/release/ -llibkame
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/debug/ -llibkame
else:unix: LIBS += -L$$OUT_PWD/ -llibkame

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/release/liblibkame.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/debug/liblibkame.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/release/libkame.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/debug/libkame.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/liblibkame.a

DEPENDPATH += $$PWD/../modules/testdriver

DESTDIR=$$OUT_PWD/../
modulefiles.files += ../modules/testdriver/libtestdriver.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/counter/libcounter.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/dcsource/libdcsource.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/dmm/libdmm.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/dso/libdso.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/flowcontroller/libflowcontroller.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/fourres/libfourres.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/funcsynth/libfuncsynth.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/levelmeter/liblevelmeter.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/lia/liblia.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/magnetps/libmagnetps.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/montecarlo/libmontecarlo.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/motor/libmotor.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/networkanalyzer/libnetworkanalyzer.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/nidaq/libnidaq.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/nmr/libnmr.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/nmr/libnmrpulser.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/sg/libsg.$${QMAKE_EXTENSION_SHLIB}
modulefiles.files += ../modules/tempcontrol/libtempcontrol.$${QMAKE_EXTENSION_SHLIB}

macx {
    modulefiles.path = Contents/MacOS/$${KAME_MODULES}
    QMAKE_BUNDLE_DATA += modulefiles
}
win32 {
    DISTFILES += modulefiles.files
}

message(Modules to be bundled: $${modulefiles.files}.)

