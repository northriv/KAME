moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libmontecarlo.la

libmontecarlo_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_srcdir)/kame/driver \
	-Iforms \
	$(all_includes)

AM_CXXFLAGS = $(CXXFLAGS) \
	-O6  \
	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

libmontecarlo_la_SOURCES = montecarlo.cpp montecarlo.h \
	interaction.cpp \
	kamemontecarlo.cpp kamemontecarlo.h
libmontecarlo_la_LIBADD = forms/libforms.la \
	@FFTW_LIBS@ 

SUBDIRS = forms
METASOURCES = AUTO
