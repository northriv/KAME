add_subdirectory(core)
    
########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/sg/core
    ${CMAKE_SOURCE_DIR}/modules/charinterface)

set(sg_SRCS
    usersignalgenerator.cpp
 )

add_library(sg MODULE ${sg_SRCS})
########### install files ###############
install(TARGETS sg LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)
