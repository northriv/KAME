moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libdmm.la

libdmm_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/modules/charinterface \
	-Iforms \
	$(all_includes) 
	
libdmm_la_SOURCES = dmm.cpp dmm.h \
	userdmm.cpp userdmm.h
libdmm_la_LIBADD = forms/libforms.la \
	../charinterface/libcharinterface.la

SUBDIRS = forms
METASOURCES = AUTO
