add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/flowcontroller/core
    )
    
########### next target ###############
set(motor_SRCS
    fujikininterface.cpp 
    userflowcontroller.cpp
 )

add_library(flowcontroller MODULE ${flowcontroller_SRCS})
target_link_libraries(flowcontroller kame_flowcontrollercore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS flowcontroller LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

