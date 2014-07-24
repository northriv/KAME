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

set(analyzer_HEADERS
	analyzer.h
	recorder.h
	recordreader.h
	)
	
QT4_WRAP_CPP(analyzer_MOC ${analyzer_HEADERS})

add_library(analyzer STATIC ${analyzer_SRCS} ${analyzer_MOC})
add_dependencies(analyzer graph driver)
########### install files ###############
