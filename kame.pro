TEMPLATE = subdirs

CONFIG += kame

SUBDIRS += kamepoolalloc\
        tests\
        kame\
        modules\

kamepoolalloc.file = kamepoolalloc/kamepoolalloc.pro

# `tests` link against `libkamepoolalloc` (built by the `kamepoolalloc`
# subdir target).  qmake doesn't automatically order subdirs, so wire
# the dependency explicitly.  In the original layout `libkame` had a
# spurious `libkame.depends = tests` line — it's preserved for now
# (the chain `kamepoolalloc → tests → libkame → modules → kame` runs
# cleanly and matches the cmake graph), but tests strictly only need
# `kamepoolalloc`.
tests.depends = kamepoolalloc

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
