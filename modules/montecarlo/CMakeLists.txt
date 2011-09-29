include_directories(
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_SOURCE_DIR}/kame/math 
    ${FFTW3_INCLUDE_DIR})
    
########### next target ###############
set(montecarlo_SRCS
    interaction.cpp
    montecarlo.cpp
    kamemontecarlo.cpp
 )

kde4_add_ui_files(montecarlo_SRCS
  montecarloform.ui  )

kde4_add_library(montecarlo MODULE ${montecarlo_SRCS})
add_dependencies(montecarlo kame)
########### install files ###############
install(TARGETS montecarlo LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
