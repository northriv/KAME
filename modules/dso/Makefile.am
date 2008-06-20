moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libdso.la

libdso_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/graph \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/modules/dso/core \
	$(all_includes)

libdso_la_SOURCES = tds.cpp tds.h \
	lecroy.cpp lecroy.h

libdso_la_LIBADD = core/libdsocore.la \
	../charinterface/libcharinterface.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

METASOURCES = AUTO

SUBDIRS = core
