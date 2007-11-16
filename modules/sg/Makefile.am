moduleexecdir=$(pkglibdir)/modules
moduleexec_LTLIBRARIES = libsg.la

libsg_la_LDFLAGS = -module

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/modules/charinterface \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/modules/sg/core \
	-Icore/forms \
	$(all_includes)

libsg_la_SOURCES = usersignalgenerator.cpp usersignalgenerator.h
libsg_la_LIBADD = core/libsgcore.la \
	../charinterface/libcharinterface.la

SUBDIRS = core
METASOURCES = AUTO
