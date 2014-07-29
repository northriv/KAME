TEMPLATE = subdirs

CONFIG += kame

unix: SUBDIRS = tests

SUBDIRS += \
        libkame\
        kame\
        modules\

libkame.file = kame/libkame.pro
unix: libkame.depends = tests
modules.depends = libkame
kame.depends = libkame
macx: kame.depends = modules
else: modules.depends += kame

TRANSLATIONS = kame_ja.ts
