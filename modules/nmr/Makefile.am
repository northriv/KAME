moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libnmrpulser.la libsg.la libnmr.la

libnmrpulser_la_LDFLAGS = -module
libsg_la_LDFLAGS = -module
libnmr_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/dso/core \
	-I$(top_srcdir)/modules/magnetps/core \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_srcdir)/modules/nmr/core \
	-Icore/forms \
	$(all_includes)

libnmrpulser_la_SOURCES = \
	pulserdriversh.cpp pulserdriversh.h \
	pulserdriverh8.cpp pulserdriverh8.h
libsg_la_SOURCES = 
libnmr_la_SOURCES = \
	nmrpulse.cpp nmrpulse.h \
	nmrspectrum.cpp nmrspectrum.h \
	nmrfspectrum.cpp nmrfspectrum.h \
	nmrrelaxfit.cpp nmrrelaxfit.h \
	nmrrelax.cpp nmrrelax.h
	
libnmrpulser_la_LIBADD = core/libnmrpulsercore.la \
	../charinterface/libcharinterface.la
libsg_la_LIBADD = core/libsgcore.la \
	../charinterface/libcharinterface.la
libnmr_la_LIBADD = core/forms/libnmrforms.la \
	../charinterface/libcharinterface.la \
	core/libsgcore.la \
	core/libnmrpulsercore.la \
	../dso/core/libdsocore.la \
	../magnetps/core/libmagnetpscore.la \
	@FFTW_LIBS@ @GSL_LIBS@

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

SUBDIRS = core
METASOURCES = AUTO
