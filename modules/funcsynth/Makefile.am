moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libfuncsynth.la

libfuncsynth_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-Iforms \
	$(all_includes)

METASOURCES = AUTO

libfuncsynth_la_SOURCES = funcsynth.cpp funcsynth.h \
	userfuncsynth.cpp userfuncsynth.h
libfuncsynth_la_LIBADD = forms/libforms.la
#	../charinterface/libcharinterface.la

SUBDIRS = forms
