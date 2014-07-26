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

win32: QMAKE_CXXFLAGS += -pie

VERSTR = '\\"4.0\\"'
DEFINES += VERSION=\"$${VERSTR}\"
KAME_MODULES = modules
DEFINES += KAME_MODULE_DIR_SURFIX=\'\"/$${KAME_MODULES}/\"\'

greaterThan(QT_MAJOR_VERSION, 4) {
}
else {
	DEFINES += DATA_INSTALL_DIR=\'\"/usr/share/kame\"\'
}

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/math\
    $${_PRO_FILE_PWD_}/forms\
    $${_PRO_FILE_PWD_}/thermometer\
    $${_PRO_FILE_PWD_}/analyzer\
    $${_PRO_FILE_PWD_}/driver\
    $${_PRO_FILE_PWD_}/graph\
    $${_PRO_FILE_PWD_}/script\
    $${_PRO_FILE_PWD_}/icons

unix {
    HEADERS += \
        allocator_prv.h \
        allocator.h
}

HEADERS += \
    kame.h \
    threadlocal.h \
    transaction_impl.h \
    transaction_signal.h \
    transaction.h \
    xthread.h \
    xtime.h \
    atomic_list.h \
    atomic_prv_x86.h \
    atomic_queue.h \
    atomic_smart_ptr.h \
    atomic.h \
    driver/driver.h \
    driver/dummydriver.h \
    driver/interface.h \
    driver/primarydriver.h \
    driver/primarydriverwiththread.h \
    driver/secondarydriver.h \
    driver/secondarydriverinterface.h \
    graph/graph.h \
    graph/graphdialogconnector.h \
    graph/graphpainter.h \
    graph/graphwidget.h \
    graph/xwavengraph.h \
    analyzer/analyzer.h \
    analyzer/recorder.h \
    analyzer/recordreader.h \
    script/xdotwriter.h \
    script/xrubysupport.h \
    script/xrubythread.h \
    script/xrubythreadconnector.h \
    script/xrubywriter.h \
    xitemnode.h \
    xlistnode.h \
    xnode.h \
    xnodeconnector_prv.h \
    xnodeconnector.h \
    xscheduler.h \
    xsignal_prv.h \
    xsignal.h \
    icons/icon.h \
    measure.h \
    support.h \
    thermometer/caltable.h \
    thermometer/thermometer.h \
    math/ar.h \
    math/cspline.h \
    math/fft.h \
    math/fir.h \
    math/freqest.h \
    math/freqestleastsquare.h \
    math/matrix.h \
    math/rand.h \
    math/spectrumsolver.h \
    forms/driverlistconnector.h \
    forms/entrylistconnector.h \
    forms/graphlistconnector.h \
    forms/interfacelistconnector.h \
    forms/nodebrowser.h \
    forms/recordreaderconnector.h \

unix: SOURCES += allocator.cpp

SOURCES += \
    icons/icon.cpp \
    icons/kame-24x24-png.c \
    xthread.cpp \
    xtime.cpp \
    support.cpp \
    graph/graphdialogconnector.cpp \
    graph/graphpainter.cpp \
    graph/graphpaintergl.cpp \
    graph/graphwidget.cpp \
    graph/xwavengraph.cpp \
    graph/graph.cpp \
    thermometer/caltable.cpp \
    thermometer/thermometer.cpp \
    xitemnode.cpp \
    xlistnode.cpp \
    xscheduler.cpp \
    xsignal.cpp \
    math/ar.cpp \
    math/cspline.cpp \
    math/fft.cpp \
    math/fir.cpp \
    math/freqest.cpp \
    math/freqestleastsquare.cpp \
    math/matrix.cpp \
    math/rand.cpp \
    math/spectrumsolver.cpp \
    script/xdotwriter.cpp \
    script/xrubysupport.cpp \
    script/xrubythread.cpp \
    script/xrubythreadconnector.cpp \
    script/xrubywriter.cpp \
    measure.cpp \
    xnode.cpp \
    xnodeconnector.cpp \
    driver/driver.cpp \
    driver/interface.cpp \
    driver/primarydriver.cpp \
    driver/secondarydriver.cpp \
    forms/driverlistconnector.cpp \
    forms/entrylistconnector.cpp \
    forms/graphlistconnector.cpp \
    forms/interfacelistconnector.cpp \
    forms/nodebrowser.cpp \
    forms/recordreaderconnector.cpp \
    analyzer/analyzer.cpp \
    analyzer/recorder.cpp \
    analyzer/recordreader.cpp\
    kame.cpp \
    main.cpp

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

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/release/ -llibkame
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/debug/ -llibkame
else:unix: LIBS += -L$$OUT_PWD/ -llibkame

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/release/liblibkame.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/debug/liblibkame.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/release/libkame.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/debug/libkame.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/liblibkame.a



macx {
    INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/1.8/Headers
    LIBS += -framework Ruby
}
else:unix {
    INCLUDEPATH += /usr/lib/ruby/1.8/i386-linux/
    LIBS += -lruby
}
win32 {
    INCLUDEPATH += "C:/Ruby187/lib/ruby/1.8/i386-mingw32/"
    LIBS += -L"C:/Ruby187/lib/" -lmsvcrt-ruby18-static
}


macx {
    INCLUDEPATH += /opt/local/include
    LIBS += -L/opt/local/lib/ #MacPorts
}
win32 {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/../../boost
    INCLUDEPATH += $${_PRO_FILE_PWD_}/../../fftw3
    INCLUDEPATH += "C:/Program Files (x86)/GnuWin32/include"
    LIBS += -L"C:/Program Files (x86)/GnuWin32/lib/" -lgsl -lgslcblas -lltdl -lz
    LIBS += -L$${_PRO_FILE_PWD_}/../../fftw3 -lfftw3-3
}

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += fftw3
unix: PKGCONFIG += gsl
unix: PKGCONFIG += zlib
unix: LIBS += -lltdl
unix: LIBS += -lclapack -lcblas -latlas

macx {
  QMAKE_LFLAGS += -all_load -dynamic
}

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

