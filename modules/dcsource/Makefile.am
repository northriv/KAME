moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libdcsource.la

libdcsource_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-Iforms \
	$(all_includes)
	
libdcsource_la_SOURCES = dcsource.cpp dcsource.h
libdcsource_la_LIBADD = forms/libforms.la \
	../charinterface/libcharinterface.la

SUBDIRS = forms
METASOURCES = AUTO
