include_directories(
    ${GLU_INCLUDE_PATH})

########### next target ###############
set(graph_SRCS
    graph.cpp
	graphdialogconnector.cpp
	graphpainter.cpp
	graphpaintergl.cpp
	graphwidget.cpp
	xwavengraph.cpp
)
##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O3 \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
set(graph_FORMS
	graphdialog.ui
	graphform.ui
	graphnurlform.ui
	)

set(graph_HEADERS
	graph.h
	graphdialogconnector.h
	graphpainter.h
	graphwidget.h
	xwavengraph.h
	)
	
QT4_WRAP_UI(graph_FORMS_HEADERS ${graph_FORMS})
QT4_WRAP_CPP(graph_MOC ${graph_HEADERS})

add_library(graph STATIC ${graph_MOC} ${graph_FORMS_HEADERS} ${graph_SRCS})

########### install files ###############