include_directories(${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_BINARY_DIR}/kame/forms
    ${CMAKE_BINARY_DIR}/kame/graph)

########### next target ###############
set(thermometer_SRCS
	caltable.cpp
	thermometer.cpp
)

kde4_add_library(thermometer STATIC ${thermometer_SRCS})
add_dependencies(thermometer forms graph)

########### install files ###############
