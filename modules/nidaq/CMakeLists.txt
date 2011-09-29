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
target_link_libraries(nidaq kame_dsocore kame_nmrpulsercore ${NIDAQMX_LIBRARY} ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS nidaq LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
