noinst_LIBRARIES = libnmr.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/nmr/forms \
	$(all_includes)

libnmr_a_SOURCES = pulserdriver.cpp pulserdriver.h \
	pulserdriverconnector.cpp pulserdriverconnector.h \
	pulserdriversh.cpp pulserdriversh.h \
	pulserdriverh8.cpp pulserdriverh8.h \
	pulserdrivernidaqmx.cpp pulserdrivernidaqmx.h \
	nmrpulse.cpp nmrpulse.h \
	nmrspectrum.cpp nmrspectrum.h \
	nmrfspectrum.cpp nmrfspectrum.h \
	nmrrelaxfit.cpp nmrrelaxfit.h \
	nmrrelax.cpp nmrrelax.h \
	signalgenerator.cpp signalgenerator.h

SUBDIRS = forms
METASOURCES = AUTO
