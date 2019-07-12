TARGET = libkame
TEMPLATE = lib

CONFIG += static

PRI_DIR = ../
include(../kame.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}\
    $${_PRO_FILE_PWD_}/graph\

FORMS += \
    graph/graphdialog.ui \
    graph/graphform.ui \
    graph/graphnurlform.ui


SOURCES +=\
    icons/kame-24x24-png.c \

HEADERS += \
