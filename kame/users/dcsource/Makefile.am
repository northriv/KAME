noinst_LIBRARIES = libdcsource.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/dcsource/forms \
	$(all_includes)
	
libdcsource_a_SOURCES = dcsource.cpp dcsource.h

SUBDIRS = forms
METASOURCES = AUTO
