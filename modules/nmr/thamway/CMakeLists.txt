include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/networkanalyzer/core
    ${CMAKE_SOURCE_DIR}/modules/sg/core
    ${CMAKE_SOURCE_DIR}/modules/nmr/pulsercore)
    
########### next target ###############
set(thamway_SRCS
	thamwayprot.cpp 
	thamwaypulser.cpp
 )
 
 kde4_add_ui_files(thamway_SRCS
  thamwayprotform.ui  )

kde4_add_library(thamway MODULE ${thamway_SRCS})

add_library(thamway MODULE ${thamway_SRCS})
target_link_libraries(thamway kame_charinterface kame_networkanalyzercore kame_sgcore kame_nmrpulsercore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS thamway LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
