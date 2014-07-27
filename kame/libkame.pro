TARGET = libkame
TEMPLATE = lib

CONFIG += static

PRI_DIR = ../
include(../kame.pri)

FORMS += \
    graph/graphdialog.ui \
    graph/graphform.ui \
    graph/graphnurlform.ui


SOURCES +=\
    icons/icon.cpp \
    icons/kame-24x24-png.c \

HEADERS += \
    icons/icon.h

