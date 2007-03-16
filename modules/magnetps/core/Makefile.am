pkglib_LTLIBRARIES = libmagnetpscore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-Iforms \
	$(all_includes)

libmagnetpscore_la_SOURCES = magnetps.cpp magnetps.h
libmagnetpscore_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
