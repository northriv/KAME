pkglib_LTLIBRARIES = libsgcore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-Iforms \
	$(all_includes)

libsgcore_la_SOURCES = \
	signalgenerator.cpp signalgenerator.h
	
libsgcore_la_LIBADD = forms/libsgforms.la

SUBDIRS = forms
METASOURCES = AUTO
