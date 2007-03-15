moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libmagnetps.la

libmagnetps_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/users/magnetps \
	-Iforms \
	$(all_includes)

libmagnetps_la_SOURCES = magnetps.cpp magnetps.h \
	usermagnetps.cpp usermagnetps.h
libmagnetps_la_LIBADD = forms/libforms.la
#	../charinterface/libcharinterface.la

SUBDIRS = forms
METASOURCES = AUTO
