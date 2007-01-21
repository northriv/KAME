noinst_LIBRARIES = libtempcontrol.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/tempcontrol/forms \
	$(all_includes)

libtempcontrol_a_SOURCES = tempcontrol.cpp tempcontrol.h \
	usertempcontrol.cpp usertempcontrol.h 

SUBDIRS = forms
METASOURCES = AUTO
