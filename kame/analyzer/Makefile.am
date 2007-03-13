noinst_LIBRARIES = libanalyzer.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_srcdir)/kame/driver \
	$(all_includes)

libanalyzer_a_SOURCES = \
	recorder.cpp recorder.h \
	recordreader.cpp recordreader.h \
	analyzer.cpp analyzer.h

METASOURCES = AUTO

