add_subdirectory(core)
    
########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/magnetps/core)

set(magnetps_SRCS
    usermagnetps.cpp
 )

add_library(magnetps MODULE ${magnetps_SRCS})
target_link_libraries(magnetps magnetpscore charinterface)
########### install files ###############
install(TARGETS magnetps LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

