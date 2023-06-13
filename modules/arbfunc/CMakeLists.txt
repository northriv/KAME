include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/modules/charinterface)
########### next target ###############
set(pumpcontroller_SRCS
	pumpcontroller.cpp 
	userpumpcontroller.cpp 
 )

kde4_add_ui_files(pumpcontroller_SRCS
  pumpcontrollerform.ui)
  
kde4_add_library(pumpcontroller MODULE ${pumpcontroller_SRCS})
target_link_libraries(pumpcontroller kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS pumpcontroller LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

