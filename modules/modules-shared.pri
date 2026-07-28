TEMPLATE = lib

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/script\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kamestm\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kamepoolalloc\


win32 {
    CONFIG += plugin

# -Wl,--whole-archive ${old_libs} -Wl,--no-whole-archive ${dependency_libs} -Wl,--enable-auto-import
}
else {
    CONFIG += shared
}

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}
win32-*g++ {
    QMAKE_LFLAGS += -Wl,--export-all-symbols
    win32-clang-g++ {
        DEFINES += DECLSPEC_KAME=__declspec(dllexport)
        DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
        DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
    }
    LIBS += $${PRI_DIR}../kame/kame.a
}
win32-msvc* {
    DEFINES += DECLSPEC_KAME=__declspec(dllimport)
    DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
    DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
    LIBS += $${PRI_DIR}../kame.lib
}
win32 {
    DESTDIR=$$OUT_PWD/$${PRI_DIR}../coremodules
    LIBS += -L$${PRI_DIR}../coremodules/
}

PRI_DIR = $${PRI_DIR}../
include(../kame.pri)

unix:!macx {
    # Linux/BSD deployment.  macOS collects the built .so files into the app
    # bundle from kame/kame.pro (QMAKE_BUNDLE_DATA) and Windows sets DESTDIR
    # in the win32 blocks above; Linux had neither, so every core module was
    # left in its own build subdirectory — a location no search path in
    # main.cpp ever looks at — and `make install` produced an empty rule.
    #
    # `unversioned_libname unversioned_soname`: these are dlopen()ed plugins,
    # not link-time libraries.  Without this, `CONFIG += shared` emits
    # libfoo.so plus three versioned symlinks, and lt_dlforeachfile() then
    # hands every one of them to lt_dlopenext().
    CONFIG += unversioned_libname unversioned_soname
    isEmpty(KAME_MODULE_TIER): KAME_MODULE_TIER = $${KAME_COREMODULES}
    DESTDIR = $$OUT_PWD/$${PRI_DIR}bin/$${KAME_MODULE_TIER}
    modulefiles.files = $${DESTDIR}/$${QMAKE_PREFIX_SHLIB}$${TARGET}.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.path = $${KAME_LIBDIR}/$${KAME_MODULE_TIER}
    modulefiles.CONFIG += no_check_exist   # built by this very Makefile
    INSTALLS += modulefiles
}

