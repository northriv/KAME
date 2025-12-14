TEMPLATE = lib

CONFIG += plugin

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\

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
        win32-clang-g++ {
            DEFINES += DECLSPEC_KAME=__declspec(dllexport)
            DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
            DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
        }
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

