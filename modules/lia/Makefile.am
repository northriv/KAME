moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = liblia.la

liblia_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_builddir)/kame/users/lia/forms \
	$(all_includes)

liblia_la_SOURCES = lockinamp.cpp lockinamp.h \
	userlockinamp.cpp userlockinamp.h
liblia_la_LIBADD = forms/libforms.la
#	../charinterface/libcharinterface.la

SUBDIRS = forms
METASOURCES = AUTO
