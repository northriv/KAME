noinst_LIBRARIES = libgraph.a

INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/kame \
	-I$(top_srcdir)/FTGL/include \
	-I$(top_builddir)/kame/graph $(all_includes)

libgraph_a_SOURCES = graph.cpp graph.h \
	graphdialogconnector.cpp graphdialogconnector.h \
	graphpainter.cpp graphpaintergl.cpp graphpainter.h \
	graphwidget.cpp graphwidget.h \
	xwavengraph.cpp xwavengraph.h \
	graphdialog.ui graphform.ui graphnurlform.ui

libgraph_a_LIBADD = 

METASOURCES = AUTO

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O3 \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
