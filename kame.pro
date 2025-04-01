TEMPLATE = subdirs

CONFIG += kame

SUBDIRS += tests\
        kame\
        modules\

macx {
    SUBDIRS += libkame #graph forms
    libkame.file = kame/libkame.pro
    libkame.depends = tests
    modules.depends = libkame
    kame.depends = modules
}
else {
    kame.depends = tests
    modules.depends = kame
}

TRANSLATIONS = kame_ja.ts
