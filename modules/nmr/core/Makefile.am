pkglib_LTLIBRARIES = libnmrcore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libnmrcore_la_SOURCES = \
	pulserdriverconnector.cpp pulserdriverconnector.h \
	pulserdriver.cpp pulserdriver.h \
	signalgenerator.cpp signalgenerator.h
	
libnmrcore_la_LIBADD = forms/libnmrpulserforms.la forms/libsgforms.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

SUBDIRS = forms
METASOURCES = AUTO
