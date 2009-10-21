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
target_link_libraries(nidaq ${NIDAQMX_LIBRARY} dsocore nmrpulsercore)
########### install files ###############
install(TARGETS nidaq LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)


#libnidaq_la_LIBADD = \
#	../dso/core/libdsocore.la \
#	../nmr/core/libnmrpulsercore.la \
#	@NIDAQMX_LIBS@
#
#METASOURCES = AUTO
