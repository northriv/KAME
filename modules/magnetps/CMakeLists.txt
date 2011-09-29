add_subdirectory(core)
    
########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/magnetps/core)

set(magnetps_SRCS
    usermagnetps.cpp
 )

add_library(magnetps MODULE ${magnetps_SRCS})
target_link_libraries(magnetps kame_magnetpscore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS magnetps LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

