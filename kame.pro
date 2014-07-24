TEMPLATE = subdirs

CONFIG += kame
CONFIG += ordered

#
SUBDIRS = tests\
        kame\
        modules\

modules.depends = kame
kame.depends = tests

#        po\

