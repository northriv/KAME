include_directories(${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_BINARY_DIR}/kame/forms
    ${CMAKE_BINARY_DIR}/kame/graph)

########### next target ###############
set(thermometer_SRCS
	caltable.cpp
	thermometer.cpp
)
set(thermometer_HEADERS
	caltable.h
	thermometer.h
	)
	
QT4_WRAP_CPP(thermometer_MOC ${thermometer_HEADERS})

add_library(thermometer STATIC ${thermometer_SRCS} ${thermometer_MOC})
add_dependencies(thermometer forms graph)

########### install files ###############
