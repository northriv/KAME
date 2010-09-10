include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_SOURCE_DIR}/modules/dso/core
    ${CMAKE_SOURCE_DIR}/modules/nmr/pulsercore)
    
########### next target ###############
set(nidaq_SRCS
	nidaqmxdriver.cpp 
	pulserdrivernidaqmx.cpp 
	pulserdrivernidaq.cpp 
	nidaqdso.cpp 
 )

add_library(nidaq MODULE ${nidaq_SRCS})
target_link_libraries(nidaq ${NIDAQMX_LIBRARY})
add_dependencies(nidaq kame)
########### install files ###############
install(TARGETS nidaq LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)
