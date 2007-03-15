moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libnmr.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/dso \
	-I$(top_srcdir)/modules/magnetps \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_builddir)/kame/users/nmr/forms \
	$(all_includes)

libnmr_la_SOURCES = \
	pulserdriverconnector.cpp pulserdriverconnector.h \
	pulserdriver.cpp pulserdriver.h \
	pulserdriversh.cpp pulserdriversh.h \
	pulserdriverh8.cpp pulserdriverh8.h \
	nmrpulse.cpp nmrpulse.h \
	nmrspectrum.cpp nmrspectrum.h \
	nmrfspectrum.cpp nmrfspectrum.h \
	nmrrelaxfit.cpp nmrrelaxfit.h \
	nmrrelax.cpp nmrrelax.h \
	signalgenerator.cpp signalgenerator.h
	
libnmr_la_LIBADD = forms/libforms.la \
	../charinterface/libcharinterface.la \
	../dso/libdso.la \
	../magnetps/libmagnetps.la \
	@FFTW_LIBS@ @GSL_LIBS@

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations

SUBDIRS = forms
METASOURCES = AUTO
