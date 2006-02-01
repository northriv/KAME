INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/funcsynth/forms \
	$(all_includes)

METASOURCES = AUTO
noinst_LIBRARIES =  libfuncsynth.a
libfuncsynth_a_SOURCES = funcsynth.cpp funcsynth.h \
	userfuncsynth.cpp userfuncsynth.h
	
SUBDIRS = forms
