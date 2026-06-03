TEMPLATE = subdirs

CONFIG += kame

SUBDIRS += kamepoolalloc\
        kamestm\
        tests\
        kame\
        modules\

kamepoolalloc.file = kamepoolalloc/kamepoolalloc.pro
kamestm.file       = kamestm/kamestm.pro

# qmake doesn't automatically order subdirs, so wire the deps
# explicitly.  The dependency chain is:
#
#   kamepoolalloc  ->  kamestm  ->  tests  ->  libkame  ->  modules  ->  kame
#
# `kamestm` depends on `kamepoolalloc` (allocator + barrier headers).
# `tests` link against both `libkamepoolalloc` and `libkamestm`.
kamestm.depends = kamepoolalloc
tests.depends   = kamepoolalloc kamestm

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
