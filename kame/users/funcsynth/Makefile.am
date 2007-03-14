INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/funcsynth/forms \
	$(all_includes)

METASOURCES = AUTO
pkglib_LTLIBRARIES =  libfuncsynth.la
libfuncsynth_la_SOURCES = funcsynth.cpp funcsynth.h \
	userfuncsynth.cpp userfuncsynth.h
libfuncsynth_la_LIBADD = forms/libforms.la

SUBDIRS = forms
