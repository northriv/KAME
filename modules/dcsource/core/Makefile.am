pkglib_LTLIBRARIES = libdcsourcecore.la

libdcsourcecore_la_LDFLAGS = 

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-Iforms \
	$(all_includes)
	
libdcsourcecore_la_SOURCES = dcsource.cpp dcsource.h
libdcsourcecore_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
