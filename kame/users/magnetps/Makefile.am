noinst_LIBRARIES = libmagnetps.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/users/magnetps \
	-I$(top_builddir)/kame/users/magnetps/forms \
	$(all_includes)

libmagnetps_a_SOURCES = magnetps.cpp magnetps.h \
	usermagnetps.cpp usermagnetps.h

SUBDIRS = forms
METASOURCES = AUTO
