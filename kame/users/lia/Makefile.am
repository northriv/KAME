pkglib_LTLIBRARIES = liblia.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/lia/forms \
	$(all_includes)

liblia_la_SOURCES = lockinamp.cpp lockinamp.h \
	userlockinamp.cpp userlockinamp.h
liblia_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
