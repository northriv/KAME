add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/motor/core
    )
    
########### next target ###############
set(motor_SRCS
    modbusrtuinterface.cpp 
    usermotor.cpp
 )

add_library(motor MODULE ${motor_SRCS})
target_link_libraries(motor kame_motorcore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS motor LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

