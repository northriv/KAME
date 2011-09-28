include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
     ${CMAKE_SOURCE_DIR}/kame/graph )
########### next target ###############
set(dsocore_SRCS
    dso.cpp 
 )

kde4_add_ui_files(dsocore_SRCS
  dsoform.ui  )

kde4_add_library(dsocore MODULE ${dsocore_SRCS})
########### install files ###############
install(TARGETS dsocore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)


##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
