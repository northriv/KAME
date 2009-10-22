include_directories(
    ${CMAKE_SOURCE_DIR}/kame/thermometer
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/dcsource/core)
########### next target ###############
set(tempcontrol_SRCS
	tempcontrol.cpp 
	usertempcontrol.cpp 
 )

kde4_add_ui_files(tempcontrol_SRCS
  tempcontrolform.ui)
  
kde4_add_library(tempcontrol MODULE ${tempcontrol_SRCS})
########### install files ###############
install(TARGETS tempcontrol LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

