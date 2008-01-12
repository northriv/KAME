pkglib_LTLIBRARIES = libdsocore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/math \
	-I$(top_srcdir)/kame/graph \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libdsocore_la_SOURCES = dso.cpp dso.h

libdsocore_la_LIBADD = forms/libforms.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

METASOURCES = AUTO

SUBDIRS = forms
