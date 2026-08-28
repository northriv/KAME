CONFIG += qt exceptions
CONFIG += rtti
contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

QT       += core gui

#remove these two to use QOpenGLWidget
#DEFINES += USE_QGLWIDGET
#QT		 += opengl

greaterThan(QT_MAJOR_VERSION, 5): QT += uitools

greaterThan(QT_MAJOR_VERSION, 5): QT += opengl openglwidgets
# NOTE: `QT += core5compat` used to be here "for QTextCodec".  Nothing in
# kame/ or modules/ uses QTextCodec, QRegExp, QStringRef or QLinkedList any
# more — the only reference left was a dead #include plus a commented-out
# setCodecForLocale() call in main.cpp (Qt 6 is UTF-8 by default, so the call
# was redundant even when it was live).  Requiring the module made
# qt6-5compat-dev a mandatory extra package on Linux, and made qmake fail
# outright ("Unknown module(s) in QT: core5compat") on any Qt 6 install that
# does not ship it.  Do not re-add it without a real user.

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
#For ruby.h
QMAKE_CXXFLAGS += -Wno-register

# --- Ruby scripting -------------------------------------------------------
# Built by default.  `qmake CONFIG+=no_ruby` drops the interpreter, its thread
# and the libruby dependency.  The .kam FORMAT is unaffected: XRubyWriter only
# writes text and is always built, and .kam files are loaded by the Python
# loader whenever USE_PYBIND11 is set, which is already the default path.
# What goes away is running .rb files and the Ruby line shell.
#
# USE_RUBY is deliberately NOT defined here.  Every target includes this file,
# but only kame/kame.pro looks for ruby.h -- so defining it here handed the
# modules a "yes" that the app itself could answer "no", and a macro that some
# translation units see and others do not is an ABI split waiting for a
# header to key a class layout on it.  8bb86a9b6 is what that costs: one
# member of XMeasure behind #ifdef USE_RUBY moved every later member by 16
# bytes, the modules' m_interfaces landed on the app's m_drivers, and adding
# any driver corrupted the node tree on the spot.  The define now lives with
# the detection, in kame/kame.pro, and only that target's own sources
# (kame.cpp, measure.cpp, kame.h) may test it.

# Run the kamepoolalloc pool allocator (NOT std::allocator) in production.
# `-= USE_STD_ALLOCATOR` clears any inherited definition so the pool path in
# allocator.h / allocator.cpp is active on every platform (macOS, Windows).
# The test suites mirror this via USE_KAME_ALLOCATOR=ON (tests/CMakeLists.txt).
# To fall back to std::allocator, add `DEFINES += USE_STD_ALLOCATOR` instead.
DEFINES -= USE_STD_ALLOCATOR


VERSTR = 8.6.1
DEFINES += VERSION=\"quotedefined($${VERSTR})\"

KAME_COREMODULES = coremodules
DEFINES += KAME_COREMODULE_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES}/)\"

KAME_COREMODULES2 = coremodules2
DEFINES += KAME_COREMODULE2_DIR_SURFIX=\"quotedefined(/$${KAME_COREMODULES2}/)\"

KAME_MODULES = modules
DEFINES += KAME_MODULE_DIR_SURFIX=\"quotedefined(/$${KAME_MODULES}/)\"

unix:!macx {
    # Linux/BSD: unlike macOS (app bundle) and Windows (modules next to the
    # .exe), there is no location that QApplication::libraryPaths() reports
    # AND that anything ever installs modules into — so a `make install`ed
    # KAME found zero drivers.  Fix both ends: install under
    # $$PREFIX/lib/kame/{coremodules,coremodules2,modules} (modules.pri /
    # modules-shared.pri) and tell main.cpp to search there (below).
    isEmpty(PREFIX): PREFIX = /usr/local
    KAME_LIBDIR = $${PREFIX}/lib/kame
    DEFINES += KAME_MODULE_INSTALL_DIR=\"quotedefined($${KAME_LIBDIR})\"
}

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
    contains(QMAKE_HOST.arch, x86_64) {
        INCLUDEPATH += c:/msys64/mingw64/include
        INCLUDEPATH += c:/msys64/mingw64/include/eigen3
        LIBS += -Lc:/msys64/mingw64/lib
    }
    else {
        INCLUDEPATH += c:/msys64/mingw32/include
        INCLUDEPATH += c:/msys64/mingw32/include/eigen3
        LIBS += -Lc:/msys64/mingw32/lib
    }
#    INCLUDEPATH += c:/msys64/usr/include
#    LIBS += -Lc:/msys64/usr/lib

    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../eigen3

    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3

#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../boost
    DEFINES += GSL_DLL
}
win32-*g++ {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/.libs
    LIBS += -lgsl #-lgslcblas
    contains(QMAKE_HOST.arch, x86_64) {
        LIBS += -lfftw3
    }
    else {
        LIBS += -lfftw3-3
    }
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
        # Linux/BSD: eigen3 is header-only and ships eigen3.pc.  macOS and
        # Windows hard-code their include path above (MacPorts / msys64);
        # this is the missing third case — without it every translation unit
        # that includes <Eigen/Core> fails to find it.
        PKGCONFIG += eigen3
    }
}

#macx: DEFINES += HAVE_LAPACK

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
    QMAKE_LFLAGS += /opt:noref
}
else {
    contains(QMAKE_HOST.arch, x86_64) {
        win32-g++ {
            #workaround for movaps alignment problem
            QMAKE_CXXFLAGS += -mstackrealign
            #increases stack size to 8MB, the same as Linux/OS X.
            QMAKE_CXXFLAGS += -Wl,--stack,8388608
            #avoids "too many sections" with Eigen.
            QMAKE_CXXFLAGS += -Wa,-mbig-obj
            #workaround for section shortage
            QMAKE_CXXFLAGS_DEBUG += -Os
        }
        win32-clang-g++ {
            #workaround for movaps alignment problem
            QMAKE_CXXFLAGS += -mstackrealign
            #increases stack size to 8MB, the same as Linux/OS X.
            QMAKE_CXXFLAGS += -Wl,--stack,8388608
        }
    }
    else {
        contains(QMAKE_HOST.arch, x86) {
            QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
            win32-g++ {
                #for stupid mingw32
                QMAKE_CXXFLAGS += -fpermissive
                #workaround for section shortage
                QMAKE_CXXFLAGS_DEBUG += -Os
                #workaround for movaps alignment problem
                QMAKE_CXXFLAGS += -mstackrealign
                #increases stack size to 8MB, the same as Linux/OS X.
                QMAKE_CXXFLAGS += -Wl,--stack,8388608
                #avoids "too many sections" with Eigen.
                QMAKE_CXXFLAGS += -Wa,-mbig-obj
            }
        }
    }
}
