TARGET = libkame
TEMPLATE = lib

CONFIG += static

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

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/math\
    $${_PRO_FILE_PWD_}/forms\
    $${_PRO_FILE_PWD_}/thermometer\
    $${_PRO_FILE_PWD_}/analyzer\
    $${_PRO_FILE_PWD_}/driver\
    $${_PRO_FILE_PWD_}/graph\
    $${_PRO_FILE_PWD_}/script\
    $${_PRO_FILE_PWD_}/icons

HEADERS += \
    kame.h \
    threadlocal.h \
    transaction_impl.h \
    transaction_signal.h \
    transaction.h \
    xthread.h \
    xtime.h \
    allocator_prv.h \
    allocator.h \
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
    pthreadlock.h \
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
    forms/recordreaderconnector.h

SOURCES += \
    kame.cpp \
    main.cpp \
    support.cpp \
    threadlocal.cpp \
    xthread.cpp \
    xtime.cpp \
    allocator.cpp \
    icons/icon.cpp \
    icons/kame-24x24-png.c \
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
    analyzer/recordreader.cpp

FORMS += \
    forms/caltableform.ui \
    forms/drivercreate.ui \
    forms/drivertool.ui \
    forms/graphtool.ui \
    forms/interfacetool.ui \
    forms/nodebrowserform.ui \
    forms/recordreaderform.ui \
    forms/rubythreadtool.ui \
    forms/scalarentrytool.ui \
    graph/graphdialog.ui \
    graph/graphform.ui \
    graph/graphnurlform.ui

macx {
    INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/1.8/Headers
}
else {
    INCLUDEPATH += /usr/lib/ruby/1.8/i386-linux/
}

macx {
    INCLUDEPATH += /opt/local/include
    DEPENDPATH += /opt/local/include
}

