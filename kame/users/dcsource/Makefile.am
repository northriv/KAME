pkglib_LTLIBRARIES = libdcsource.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/dcsource/forms \
	$(all_includes)
	
libdcsource_la_SOURCES = dcsource.cpp dcsource.h
libdcsource_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
