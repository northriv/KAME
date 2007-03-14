pkglib_LTLIBRARIES = libtempcontrol.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/tempcontrol/forms \
	$(all_includes)

libtempcontrol_la_SOURCES = tempcontrol.cpp tempcontrol.h \
	usertempcontrol.cpp usertempcontrol.h 
libtempcontrol_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
