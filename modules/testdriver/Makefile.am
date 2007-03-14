moduleexecdir=$(pkglibdir)/module
moduleexec_LTLIBRARIES = libtestdriver.la

INCLUDES = $(all_includes) -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/tempcontrol/forms

libtestdriver_la_SOURCES = testdriver.cpp testdriver.h

METASOURCES = AUTO
