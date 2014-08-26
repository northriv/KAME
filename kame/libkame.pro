TARGET = libkame
TEMPLATE = lib

CONFIG += static

PRI_DIR = ../
include(../kame.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}\

FORMS += \
    graph/graphdialog.ui \
    graph/graphform.ui \
    graph/graphnurlform.ui


SOURCES +=\
    icons/icon.cpp \
    icons/kame-24x24-png.c \

HEADERS += \
    icons/icon.h

win32-msvc* {
    DEFINES += DECLSPEC_KAME=__declspec(dllexport)
    DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
    DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
}
