include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
     ${CMAKE_SOURCE_DIR}/kame/graph
     ${CMAKE_SOURCE_DIR}/modules/sg/core
)
########### next target ###############
set(dsocore_SRCS
    dso.cpp 
 )

kde4_add_ui_files(dsocore_SRCS
  dsoform.ui  )

kde4_add_library(kame_dsocore SHARED ${dsocore_SRCS})
add_dependencies(kame_dsocore kame)
target_link_libraries(kame_dsocore kame_sgcore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS kame_dsocore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})


##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
