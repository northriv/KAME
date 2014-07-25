TEMPLATE = subdirs

CONFIG += kame
CONFIG += ordered

unix: SUBDIRS = tests

SUBDIRS += \
        libkame\
        modules\
        kame\

libkame.file = kame/libkame.pro
unix: libkame.depends = tests
modules.depends = libkame
kame.depends = modules libkame

