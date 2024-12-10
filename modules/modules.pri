TEMPLATE = lib

CONFIG += plugin

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/script\

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32 {
    DESTDIR=$$OUT_PWD/$${PRI_DIR}
    win32-msvc* {
        DEFINES += DECLSPEC_KAME=__declspec(dllimport)
        DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
        DEFINES += DECLSPEC_SHARED=__declspec(dllimport)
        LIBS += $${PRI_DIR}../kame.lib
    }
    else {
        QMAKE_LFLAGS += -Wl,--export-all-symbols
    # -Wl,--whole-archive ${old_libs} -Wl,--no-whole-archive ${dependency_libs} -Wl,--enable-auto-import
        LIBS += $${PRI_DIR}../kame/kame.a
    }
}

unix {
    modulefiles.files = $${TARGET}.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.path = $$[QT_INSTALL_LIBS]/$${KAME_MODULES}
    INSTALLS += modulefiles
}

win32: LIBS += -L$${PRI_DIR}../coremodules/
win32: LIBS += -L$${PRI_DIR}../coremodules2/

PRI_DIR = $${PRI_DIR}../
include(../kame.pri)

#pybind11
macx {
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="python3" $$files("/opt/local/bin/python3*") $$files("/usr/local/bin/python3*")
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
#                QMAKE_LFLAGS += $$system("$${PYTHON}-config --embed --ldflags")
                DEFINES += USE_PYBIND11
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
win32-g++ {
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="c:/msys64/mingw64/bin/python.exe"
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
        #        QMAKE_LFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --embed --ldflags\"")
#                LIBS += $$files(c:/msys64/mingw64/lib/libpython3*)
                DEFINES += USE_PYBIND11
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
