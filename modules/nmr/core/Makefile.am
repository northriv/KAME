pkglib_LTLIBRARIES = libnmrpulsercore.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/math \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-Iforms \
	$(all_includes)

libnmrpulsercore_la_SOURCES = \
	pulserdriverconnector.cpp pulserdriverconnector.h \
	pulserdriver.cpp pulserdriver.h
	
libnmrpulsercore_la_LIBADD = forms/libnmrpulserforms.la

SUBDIRS = forms
METASOURCES = AUTO
