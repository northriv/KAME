####### kdevelop will overwrite this part!!! (begin)##########
noinst_LIBRARIES = libmontecarlo.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_srcdir)/kame/driver \
	-I$(top_builddir)/kame/users/montecarlo/forms \
	$(all_includes)

libmontecarlo_a_METASOURCES = AUTO

libmontecarlo_a_CXXFLAGS = $(CXXFLAGS) \
	-O6 -march=prescott -mfpmath=sse -msse3 -mmmx -funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload

libmontecarlo_a_SOURCES = montecarlo.cpp montecarlo.h \
	interaction.cpp \
	kamemontecarlo.cpp kamemontecarlo.h

####### kdevelop will overwrite this part!!! (end)############
SUBDIRS = forms
METASOURCES = AUTO
