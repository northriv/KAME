pkglib_LTLIBRARIES = libnmrpulsercore.la libsgcore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libnmrpulsercore_la_SOURCES = \
	pulserdriverconnector.cpp pulserdriverconnector.h \
	pulserdriver.cpp pulserdriver.h
libsgcore_la_SOURCES = \
	signalgenerator.cpp signalgenerator.h
	
libnmrpulsercore_la_LIBADD = forms/libnmrpulserforms.la
libsgcore_la_LIBADD = forms/libsgforms.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

SUBDIRS = forms
METASOURCES = AUTO
