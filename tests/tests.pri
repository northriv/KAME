TEMPLATE = app

CONFIG += exceptions
CONFIG += rtti
contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
    CONFIG += sse sse2
}

CONFIG += c++17

CONFIG += console
CONFIG += testcase
CONFIG -= app_bundle #macosx

INCLUDEPATH += $${_PRO_FILE_PWD_}/../kame

# Preinclude standalone support header to block Qt-dependent originals
QMAKE_CXXFLAGS += -include $${_PRO_FILE_PWD_}/support_standalone.h

win32-msvc* {
    QMAKE_CXXFLAGS += /arch:SSE2
}
else {
    contains(QMAKE_HOST.arch, x86) {
        QMAKE_CXXFLAGS += -mfpmath=sse -msse -msse2
    }
}
