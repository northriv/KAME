pkglib_LTLIBRARIES = libdmmcore.la

libdmmcore_la_LDFLAGS =

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-Iforms \
	$(all_includes) 
	
libdmmcore_la_SOURCES = dmm.cpp dmm.h 
libdmmcore_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
