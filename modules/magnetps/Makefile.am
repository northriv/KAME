moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libmagnetps.la

libmagnetps_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/modules/magnetps/core \
	$(all_includes)

libmagnetps_la_SOURCES = \
	usermagnetps.cpp usermagnetps.h
libmagnetps_la_LIBADD = core/libmagnetpscore.la \
	../charinterface/libcharinterface.la

SUBDIRS = core
METASOURCES = AUTO
