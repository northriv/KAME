pkglib_LTLIBRARIES = libnmr.la

SUBDIRS = forms
METASOURCES = AUTO

INCLUDES = -I$(top_srcdir)/kame \
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
	pulserdrivernidaq.cpp pulserdrivernidaq.h \
	pulserdrivernidaqmx.cpp pulserdrivernidaqmx.h \
	nmrpulse.cpp nmrpulse.h \
	nmrspectrum.cpp nmrspectrum.h \
	nmrfspectrum.cpp nmrfspectrum.h \
	nmrrelaxfit.cpp nmrrelaxfit.h \
	nmrrelax.cpp nmrrelax.h \
	signalgenerator.cpp signalgenerator.h
	
libnmr_la_LIBADD = forms/libforms.la \
	../dso/libdso.la \
	../magnetps/libmagnetps.la

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
