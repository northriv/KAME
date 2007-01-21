####### kdevelop will overwrite this part!!! (begin)##########
noinst_LIBRARIES = libgraph.a

INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/kame \
	-I$(top_srcdir)/FTGL/include \
	-I$(top_builddir)/kame/graph/forms $(all_includes)

libgraph_a_SOURCES = graph.cpp graph.h \
	graphdialogconnector.cpp graphdialogconnector.h \
	graphpainter.cpp graphpaintergl.cpp graphpainter.h \
	graphwidget.cpp graphwidget.h \
	xwavengraph.cpp xwavengraph.h 
	
METASOURCES = AUTO

SUBDIRS = forms

noinst_HEADERS = graphpainter.h
