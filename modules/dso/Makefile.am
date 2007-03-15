moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libdso.la

libdso_la_LDFLAGS = -module -L$(moduleexecdir)\
	-dlpreopen=libcharinterface.so

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/graph \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libdso_la_SOURCES = fir.cpp fir.h \
	dso.cpp dso.h \
	tds.cpp tds.h
libdso_la_LIBADD = forms/libforms.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

METASOURCES = AUTO

SUBDIRS = forms


