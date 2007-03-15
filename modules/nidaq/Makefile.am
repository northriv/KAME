moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libnidaq.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/dso \
	-I$(top_srcdir)/modules/nmr \
	$(all_includes)

libnidaq_la_SOURCES = \
	nidaqmxdriver.cpp nidaqmxdriver.h \
	pulserdrivernidaq.cpp pulserdrivernidaq.h \
	nidaqdso.cpp nidaqdso.h
	
libnidaq_la_LIBADD = \
	../dso/libdso.la \
	../nmr/libnmr.la \
	@NIDAQMX_LIBS@

METASOURCES = AUTO
