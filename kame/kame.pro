TARGET = kame
TEMPLATE = app

PRI_DIR = ../
include(../kame.pri)

macx: SCRIPT_DIR = Resources
win32: SCRIPT_DIR = resources
DEFINES += LINESHELL_DIR=\"quotedefined($${SCRIPT_DIR}/)\"
DEFINES += USE_STD_RANDOM

CONFIG += CONSOLE

#win32: QMAKE_CXXFLAGS += -pie

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}\
    $${_PRO_FILE_PWD_}/math\
    $${_PRO_FILE_PWD_}/forms\
    $${_PRO_FILE_PWD_}/thermometer\
    $${_PRO_FILE_PWD_}/analyzer\
    $${_PRO_FILE_PWD_}/driver\
    $${_PRO_FILE_PWD_}/graph\
    $${_PRO_FILE_PWD_}/script\
    $${_PRO_FILE_PWD_}/icons

HEADERS += \
    allocator.h \
    atomic_prv_mfence_arm8.h \
    graph/graphmathfittool.h \
    graph/graphmathtool.h \
    graph/graphmathtoolconnector.h \
    graph/graphntoolbox.h \
    graph/onscreenobject.h \
    graph/x2dimage.h \
    kame.h \
    script/xscriptingthread.h \
    script/xscriptingthreadconnector.h \
    threadlocal.h \
    transaction_impl.h \
    transaction_signal.h \
    transaction.h \
    xthread.h \
    xtime.h \
    atomic_prv_std.h \
    atomic_prv_basic.h \
    atomic_prv_mfence_x86.h \
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
    driver/softtrigger.h \
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
    script/xrubywriter.h \
    script/rubywrapper.h \
    xitemnode.h \
    xlistnode.h \
    xnode.h \
    xnodeconnector.h \
    xscheduler.h \
    icons/icon.h \
    measure.h \
    support.h \
    thermometer/caltable.h \
    thermometer/thermometer.h \
    math/ar.h \
    math/cspline.h \
    math/fft.h \
    math/fir.h \
    math/freqestleastsquare.h \
    math/rand.h \
    math/spectrumsolver.h \
    forms/driverlistconnector.h \
    forms/entrylistconnector.h \
    forms/graphlistconnector.h \
    forms/interfacelistconnector.h \
    forms/nodebrowser.h \
    forms/recordreaderconnector.h \
    messagebox.h \
    math/nllsfit.h \
    math/tikhonovreg.h

SOURCES += icons/icon.cpp \
    graph/graphmathtool.cpp \
    graph/graphmathtoolconnector.cpp \
    graph/graphntoolbox.cpp \
    graph/onscreenobject.cpp \
    graph/x2dimage.cpp \
    script/xscriptingthread.cpp \
    script/xscriptingthreadconnector.cpp \
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
    math/ar.cpp \
    math/cspline.cpp \
    math/fft.cpp \
    math/fir.cpp \
    math/freqestleastsquare.cpp \
    math/rand.cpp \
    math/spectrumsolver.cpp \
    script/xdotwriter.cpp \
    script/xrubysupport.cpp \
    script/xrubywriter.cpp \
    script/rubywrapper.cpp \
    measure.cpp \
    xnode.cpp \
    xnodeconnector.cpp \
    driver/driver.cpp \
    driver/interface.cpp \
    driver/primarydriver.cpp \
    driver/secondarydriver.cpp \
    driver/softtrigger.cpp \
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
    main.cpp \
    messagebox.cpp \
    math/tikhonovreg.cpp

unix {
    HEADERS += \
        math/matrix.h \
        math/freqest.h
    SOURCES += \
        math/freqest.cpp \
        math/matrix.cpp \
        allocator_prv.h \
        allocator.cpp \
}

FORMS += \
    forms/caltableform.ui \
    forms/drivercreate.ui \
    forms/drivertool.ui \
    forms/graphtool.ui \
    forms/interfacetool.ui \
    forms/nodebrowserform.ui \
    forms/recordreaderform.ui \
    forms/scalarentrytool.ui \
    forms/messageform.ui \
    forms/scriptingthreadtool.ui

RESOURCES += \
    kame.qrc

DESTDIR=$$OUT_PWD/../

scriptfile.files = script/rubylineshell.rb \
    script/pythonlineshell.py \
    script/notebook/jupyter_notebook_config.py \
    script/notebook/notebook_kame_kernel_manager.py

macx {
    scriptfile.path = Contents/Resources
    QMAKE_BUNDLE_DATA += scriptfile

    LIBS += -L$$OUT_PWD/ -llibkame
}
else {
    #in macx, these are in libkame
    FORMS += \
        graph/graphdialog.ui \
        graph/graphform.ui \
        graph/graphnurlform.ui
    SOURCES +=\
        icons/kame-24x24-png.c

    unix {
        INSTALLS += scriptfile
    }
    else {
        DISTFILES += script/rubylineshell.rb  \
            script/pythonlineshell.py \
            script/notebook/jupyter_notebook_config.py \
            script/notebook/notebook_kame_kernel_manager.py
    }
}

#win32: QMAKE_POST_LINK += $$quote(cmd /c copy /y $${_PRO_FILE_PWD_}$${scriptfile.files} $${DESTDIR}$${SCRIPT_DIR}$$escape_expand(\\n\\t))

macx: ICON = kame.icns

#Ruby, pybind11
macx {
    exists("/opt/local/include/ruby-*") {
        #for macports ruby3
        RUBYH = $$files("/opt/local/include/ruby-*")
        INCLUDEPATH += $${RUBYH}
        INCLUDEPATH += $${RUBYH}/arm64-darwin23
        LIBS += $$files(/opt/local/lib/libruby.*.dylib)
        message("using ruby from macports.")
    }
    else {
        INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/Current/Headers
        INCLUDEPATH += /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/
        LIBS += -framework Ruby
    #for ruby.h incompatible with C++11
        QMAKE_CXXFLAGS += -Wno-error=reserved-user-defined-literal
        message("using framework ruby.")
    }

    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="python3" $$files("/opt/local/bin/python3*") $$files("/usr/local/bin/python3*")
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
    #            QMAKE_CXXFLAGS += $$system("$${PYTHON}-config --cflags")
                QMAKE_LFLAGS += $$system("$${PYTHON}-config --embed --ldflags")
                DEFINES += USE_PYBIND11
                DEFINES += PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF #For mainthread call.
                SOURCES += script/xpythonmodule.cpp \
                    script/xpythonsupport.cpp
                HEADERS += script/xpythonmodule.h \
                    script/xpythonsupport.h \
                    driver/pythondriver.h
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
else:unix {
    INCLUDEPATH += /usr/lib/ruby/1.8/i386-linux/
    LIBS += -lruby
}
win32-*g++ {
    exists($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include/ruby.h) {
    #for user-build ruby
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/i386-mingw32
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/x64-mingw64
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/x64-mingw32
        LIBS += $$files($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/lib*msvcrt-ruby*[0-9].dll.a)
        message("using ruby from ../ruby.")
    }
    else {
    #for msys64 ruby
        RUBYH = $$files("c:/msys64/mingw64/include/ruby-*")
        INCLUDEPATH += $${RUBYH}
        INCLUDEPATH += $${RUBYH}/x64-mingw32
        LIBS += $$files(c:/msys64/mingw64/lib/libx64-msvcrt-ruby*[0-9].dll.a)
        message("using ruby from msys2.")
    }
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="c:/msys64/mingw64/bin/python.exe"
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
    #            QMAKE_CXXFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --cflags\"")
        #        QMAKE_LFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --embed --ldflags\"")
                LIBS += $$files(c:/msys64/mingw64/lib/libpython3*)
                DEFINES += USE_PYBIND11
                DEFINES += PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF #For mainthread call.
                SOURCES += script/xpythonmodule.cpp \
                    script/xpythonsupport.cpp
                HEADERS += script/xpythonmodule.h \
                    script/xpythonsupport.h \
                    driver/pythondriver.h
                message("Python scripting support enabled.")
                break()
            }
        }
    }
    LIBS += -lopengl32 -lglu32
}
win32-msvc* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/i386-mswin32_120
    !exists($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/libmsvcr*-ruby2*[0-9].lib) {
        error("No Ruby2 library!")
    }
    LIBS += $$files($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/libmsvcr*-ruby2*[0-9].lib)
#    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby -lmsvcr120-ruby212 #-static -lWS2_32 -lAdvapi32 -lShell32 -limagehlp -lShlwapi -lIphlpapi
}

win32 {
    contains(QMAKE_HOST.arch, x86_64) {
        LIBS += -lz
    }
    else {
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../zlib/include
        LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../zlib/lib
        LIBS += -lzdll
    }
}
win32-msvc* {
    QMAKE_PRE_LINK += lib /machine:x86 /def:$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3/libfftw3-3.def /out:$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3/libfftw3-3.lib
    QMAKE_PRE_LINK += & lib /machine:x86 /def:$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/libgsl.def /out:$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/libgsl.lib
}

unix {
#    LIBS += -lclapack -lcblas -latlas
    macx {
        LIBS += -lfftw3
        LIBS += -lz
    }
    else {
        PKGCONFIG += fftw3
        PKGCONFIG += zlib
    }
    LIBS += -lltdl
}

#exports symbols from the executable for plugins.
macx {
  QMAKE_LFLAGS += -all_load -dynamic
}
win32-*g++ {
  QMAKE_LFLAGS += -Wl,--export-all-symbols -Wl,--out-implib,$${TARGET}.a
  QMAKE_CXXFLAGS += -fvisibility=hidden
}
win32-msvc* {
    DEFINES += DECLSPEC_KAME=__declspec(dllexport)
    DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
    DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
}

macx {
    HEADERS += \
        support_osx.h

    OBJECTIVE_SOURCES += \
        support_osx.mm

    LIBS += -framework Foundation

    coremodulefiles.files += ../modules/charinterface/libcharinterface.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/dcsource/core/libdcsourcecore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/dmm/core/libdmmcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/flowcontroller/core/libflowcontrollercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/levelmeter/core/liblevelmetercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/magnetps/core/libmagnetpscore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/motor/core/libmotorcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/networkanalyzer/core/libnetworkanalyzercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/nmr/pulsercore/libnmrpulsercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/sg/core/libsgcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/lia/core/libliacore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/dso/core/libdsocore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/qd/core/libqdcore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/optics/core/libopticscore.$${QMAKE_EXTENSION_SHLIB}
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
    modulefiles.files += ../modules/nmr/thamway/libthamway.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/qd/libqd.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/digilentwf/libdigilentwf.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/gauge/libgauge.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/pumpcontroller/libpumpcontroller.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/arbfunc/libarbfunc.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/optics/liboptics.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/twoaxis/libtwoaxis.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/python/libpython.$${QMAKE_EXTENSION_SHLIB}

    coremodulefiles.path = Contents/MacOS/$${KAME_COREMODULES}
    QMAKE_BUNDLE_DATA += coremodulefiles
    coremodule2files.path = Contents/MacOS/$${KAME_COREMODULES2}
    QMAKE_BUNDLE_DATA += coremodule2files
    modulefiles.path = Contents/MacOS/$${KAME_MODULES}
    QMAKE_BUNDLE_DATA += modulefiles

    tsfiles.files += ../kame_ja.qm
    tsfiles.path = Contents/MacOS/
    QMAKE_BUNDLE_DATA += tsfiles

    QMAKE_INFO_PLIST = ../Info.plist

#    exists("/opt/local/include/libusb-1.0/libusb.h") {
    exists("../modules/nmr/thamway/fx2fw.bix") {
        ezusbfiles.path = Contents/Resources
        ezusbfiles.files += ../modules/nmr/thamway/fx2fw.bix
        ezusbfiles.files += ../modules/nmr/thamway/slow_dat.bin
        ezusbfiles.files += ../modules/nmr/thamway/fullspec_dat.bin
        QMAKE_BUNDLE_DATA += ezusbfiles
    }
}


