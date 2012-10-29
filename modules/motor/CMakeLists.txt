add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/networkanalyzer/core
    ${CMAKE_SOURCE_DIR}/modules/motor/core
    )
    
########### next target ###############
set(motor_SRCS
    modbusrtuinterface.cpp 
    usermotor.cpp
    autolctuner.cpp
 )
 
kde4_add_ui_files(motor_SRCS
	autolctunerform.ui  )
  
add_library(motor MODULE ${motor_SRCS})
target_link_libraries(motor kame_motorcore kame_networkanalyzercore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS motor LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

