pkglib_LTLIBRARIES = libmagnetps.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/users/magnetps \
	-I$(top_builddir)/kame/users/magnetps/forms \
	$(all_includes)

libmagnetps_la_SOURCES = magnetps.cpp magnetps.h \
	usermagnetps.cpp usermagnetps.h
libmagnetps_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
