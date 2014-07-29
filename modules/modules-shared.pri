TEMPLATE = lib

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/analyzer\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/driver\
    $${_PRO_FILE_PWD_}/$${PRI_DIR}../kame/math\


win32 {
    CONFIG += plugin

    QMAKE_LFLAGS += -Wl,--export-all-symbols
# -Wl,--whole-archive ${old_libs} -Wl,--no-whole-archive ${dependency_libs} -Wl,--enable-auto-import

    LIBS += $${PRI_DIR}../kame/kame.a
}
else {
    CONFIG += shared
}

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32 {
    DESTDIR=$$OUT_PWD/$${PRI_DIR}../coremodules
    LIBS += -L$${PRI_DIR}../coremodules/
}

PRI_DIR = $${PRI_DIR}../
include(../kame.pri)

