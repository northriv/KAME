moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libnidaq.la

libnidaq_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_srcdir)/modules/dso/core \
	-I$(top_srcdir)/modules/nmr/core \
	$(all_includes)

libnidaq_la_SOURCES = \
	nidaqmxdriver.cpp nidaqmxdriver.h \
	pulserdrivernidaqmx.cpp pulserdrivernidaqmx.h \
	pulserdrivernidaq.cpp pulserdrivernidaq.h \
	nidaqdso.cpp nidaqdso.h
	
libnidaq_la_LIBADD = \
	../dso/core/libdsocore.la \
	../nmr/core/libnmrpulsercore.la \
	@NIDAQMX_LIBS@

METASOURCES = AUTO
