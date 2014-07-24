include_directories(
    ${FTGL_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIR}
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

kde4_add_ui_files(graph_SRCS
  graphdialog.ui graphform.ui graphnurlform.ui  )

kde4_add_library(graph STATIC ${graph_SRCS})

########### install files ###############