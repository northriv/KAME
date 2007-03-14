moduleexecdir=$(pkglibdir)/module
moduleexec_LTLIBRARIES = libnidaq.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	$(all_includes)

libnidaq_la_SOURCES = \
	nidaqmxdriver.cpp nidaqmxdriver.h

METASOURCES = AUTO
