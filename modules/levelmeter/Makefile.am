moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = liblevelmeter.la

liblevelmeter_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/modules/levelmeter/core \
	$(all_includes)

liblevelmeter_la_SOURCES = \
	userlevelmeter.cpp userlevelmeter.h
liblevelmeter_la_LIBADD = core/liblevelmetercore.la \
	../charinterface/libcharinterface.la

SUBDIRS = core
METASOURCES = AUTO
