noinst_LIBRARIES = liblia.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/lia/forms \
	$(all_includes)

liblia_a_SOURCES = lockinamp.cpp lockinamp.h \
	userlockinamp.cpp userlockinamp.h

SUBDIRS = forms
METASOURCES = AUTO
