TEMPLATE = subdirs

CONFIG += kame

SUBDIRS += tests\
        ruby\
        libkame\
        kame\
        modules\

ruby.file = kame/script/rubywrapper.pro
libkame.file = kame/libkame.pro
libkame.depends = tests
modules.depends = libkame
kame.depends = ruby libkame
macx: kame.depends = modules
else: modules.depends += kame

TRANSLATIONS = kame_ja.ts
