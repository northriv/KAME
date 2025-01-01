TEMPLATE = subdirs

CONFIG += kame

SUBDIRS += tests\
        libkame\
        kame\
        modules\

libkame.file = kame/libkame.pro
libkame.depends = tests
modules.depends = libkame
kame.depends = libkame
macx: kame.depends += modules
else: modules.depends += kame

TRANSLATIONS = kame_ja.ts
