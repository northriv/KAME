TEMPLATE = lib

# Tell libkame headers (kame/threadlocal.h) that this TU is being
# compiled as a (shared) plugin module — XThreadLocal::operator*()
# uses a per-DLL per-thread cached pointer instead of calling
# libkame's `libkame_storage()` on every access.
DEFINES += BUILDING_PLUGIN

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/script\


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

