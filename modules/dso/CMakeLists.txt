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
target_link_libraries(dso kame_dsocore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS dso LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
#
#METASOURCES = AUTO
#
#SUBDIRS = core
