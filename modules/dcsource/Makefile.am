moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libdcsource.la

libdcsource_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/modules/dcsource/core \
	$(all_includes)
	
libdcsource_la_SOURCES = userdcsource.cpp userdcsource.h
libdcsource_la_LIBADD = core/libdcsourcecore.la \
	../charinterface/libcharinterface.la

SUBDIRS = core
METASOURCES = AUTO
