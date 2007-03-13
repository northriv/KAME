pkglib_LTLIBRARIES = libgraph.la

libgraph_la_LDFLAGS = -version-info 0:0:0
libgraph_la_LIBADD =  forms/libforms.a

INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/kame \
	-I$(top_srcdir)/FTGL/include \
	-I$(top_builddir)/kame/graph/forms $(all_includes)

libgraph_la_SOURCES = graph.cpp graph.h \
	graphdialogconnector.cpp graphdialogconnector.h \
	graphpainter.cpp graphpaintergl.cpp graphpainter.h \
	graphwidget.cpp graphwidget.h \
	xwavengraph.cpp xwavengraph.h

METASOURCES = AUTO

SUBDIRS = forms

AM_CXXFLAGS = $(CXXFLAGS) \
	-O3 \
	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
