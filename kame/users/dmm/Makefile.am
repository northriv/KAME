noinst_LIBRARIES = libdmm.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/dmm/forms \
	$(all_includes)

libdmm_a_METASOURCES = AUTO

libdmm_a_SOURCES = dmm.cpp dmm.h \
	userdmm.cpp userdmm.h

SUBDIRS = forms
METASOURCES = AUTO
