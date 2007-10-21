moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libnetworkanalyzer.la

libnetworkanalyzer_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libnetworkanalyzer_la_SOURCES = \
	networkanalyzer.cpp networkanalyzer.h \
	usernetworkanalyzer.cpp usernetworkanalyzer.h
	
libnetworkanalyzer_la_LIBADD = forms/libnetworkanalyzerform.la \
	../charinterface/libcharinterface.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

SUBDIRS = forms
METASOURCES = AUTO
