TEMPLATE = lib

CONFIG += plugin

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kamestm\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kamepoolalloc\

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32 {
    DESTDIR=$$OUT_PWD/$${PRI_DIR}
    win32-msvc* {
        DEFINES += DECLSPEC_KAME=__declspec(dllimport)
        DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
        DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
        LIBS += $${PRI_DIR}../kame.lib
    }
    else {
        QMAKE_LFLAGS += -Wl,--export-all-symbols
        win32-clang-g++ {
            DEFINES += DECLSPEC_KAME=__declspec(dllimport)
            DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
            DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
        }
    # -Wl,--whole-archive ${old_libs} -Wl,--no-whole-archive ${dependency_libs} -Wl,--enable-auto-import
        LIBS += $${PRI_DIR}../kame/kame.a
    }
}

win32: LIBS += -L$${PRI_DIR}../coremodules/
win32: LIBS += -L$${PRI_DIR}../coremodules2/

PRI_DIR = $${PRI_DIR}../
include(../kame.pri)

unix:!macx {
    # This block used to sit ABOVE `include(../kame.pri)` and read
    #     modulefiles.files = $${TARGET}.$${QMAKE_EXTENSION_SHLIB}
    #     modulefiles.path  = $$[QT_INSTALL_LIBS]/$${KAME_MODULES}
    # Both were broken: KAME_MODULES is defined by kame.pri and so was still
    # empty (the path collapsed to Qt's own libdir), and `.files` named
    # `dmm.so` while the artefact is `libdmm.so`, which made qmake emit an
    # install rule with no commands at all.  It also never set DESTDIR, so the
    # leaf modules stayed scattered one-per-build-subdirectory where no module
    # search path could find them.  See modules-shared.pri for the core-module
    # counterpart.
    DESTDIR = $$OUT_PWD/$${PRI_DIR}bin/$${KAME_MODULES}
    modulefiles.files = $${DESTDIR}/$${QMAKE_PREFIX_SHLIB}$${TARGET}.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.path = $${KAME_LIBDIR}/$${KAME_MODULES}
    modulefiles.CONFIG += no_check_exist   # built by this very Makefile
    INSTALLS += modulefiles
}

