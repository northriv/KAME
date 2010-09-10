include_directories(
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_BINARY_DIR}/kame/graph
    ${CMAKE_SOURCE_DIR}/kame/driver
    ${ZLIB_INCLUDE_DIR})

########### next target ###############
set(analyzer_SRCS
    recorder.cpp
	recordreader.cpp
	analyzer.cpp)

kde4_add_library(analyzer STATIC ${analyzer_SRCS})
add_dependencies(analyzer graph driver)
########### install files ###############
