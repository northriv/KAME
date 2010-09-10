add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/kame/graph 
    ${CMAKE_SOURCE_DIR}/modules/dso/core)

########### next target ###############
set(dso_SRCS
    tds.cpp
    lecroy.cpp 
 )

add_library(dso MODULE ${dso_SRCS})
add_dependencies(dso kame)
########### install files ###############
install(TARGETS dso LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
#
#METASOURCES = AUTO
#
#SUBDIRS = core
