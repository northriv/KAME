####### kdevelop will overwrite this part!!! (begin)##########
noinst_LIBRARIES = libdso.a

INCLUDES = -I$(top_srcdir)/. \
	-I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/graph \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/users/nmr \
	-I$(top_builddir)/kame/graph/forms \
	-I$(top_builddir)/kame/users/dso/forms \
	$(all_includes)

libdso_a_SOURCES = fir.cpp fir.h \
	nidaqdso.cpp nidaqdso.h \
	dso.cpp dso.h \
	tds.cpp tds.h

METASOURCES = AUTO

SUBDIRS = forms


