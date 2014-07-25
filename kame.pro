TEMPLATE = subdirs

CONFIG += kame
CONFIG += ordered

#
SUBDIRS = tests\
        libkame\
        modules\
        kame\

libkame.file = kame/libkame.pro
libkame.depends = tests
modules.depends = libkame
kame.depends = modules libkame

