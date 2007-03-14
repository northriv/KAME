pkglib_LTLIBRARIES = libdmm.la

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/dmm/forms \
	$(all_includes)

libdmm_la_SOURCES = dmm.cpp dmm.h \
	userdmm.cpp userdmm.h
libdmm_la_LIBADD = forms/libforms.la

SUBDIRS = forms
METASOURCES = AUTO
