include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/thermometer
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/dcsource/core
    ${CMAKE_SOURCE_DIR}/modules/flowcontroller/core)
########### next target ###############
set(tempcontrol_SRCS
	tempcontrol.cpp 
	usertempcontrol.cpp 
 )

kde4_add_ui_files(tempcontrol_SRCS
  tempcontrolform.ui)
  
kde4_add_library(tempcontrol MODULE ${tempcontrol_SRCS})
target_link_libraries(tempcontrol kame_dcsourcecore kame_flowcontrollercore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS tempcontrol LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

