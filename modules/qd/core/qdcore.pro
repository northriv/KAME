PRI_DIR = ../../
# Depends on another core module, so it must load after coremodules/.
KAME_MODULE_TIER = coremodules2
include($${PRI_DIR}/modules-shared.pri)

HEADERS += \
    qdppms.h \

SOURCES += \
    qdppms.cpp \

FORMS += \
    qdppmsform.ui

