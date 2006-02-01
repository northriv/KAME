####### kdevelop will overwrite this part!!! (begin)##########
noinst_LIBRARIES = libanalyzer.a

INCLUDES = -I$(top_srcdir)/kame -I$(top_srcdir)/kame/analyzer \
	-I$(top_builddir)/kame/analyzer/forms \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph/forms \
	$(all_includes)

libanalyzer_a_METASOURCES = AUTO

libanalyzer_a_SOURCES = recordreaderconnector.cpp recordreaderconnector.h \
	graphlistconnector.cpp graphlistconnector.h \
	entrylistconnector.cpp entrylistconnector.h \
	recorder.cpp recorder.h \
	recordreader.cpp recordreader.h \
	analyzer.cpp analyzer.h

####### kdevelop will overwrite this part!!! (end)############
SUBDIRS = forms
METASOURCES = AUTO

