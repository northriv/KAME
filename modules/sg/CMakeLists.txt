add_subdirectory(core)
    
########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/sg/core
    ${CMAKE_SOURCE_DIR}/modules/charinterface)

set(sg_SRCS
    usersignalgenerator.cpp
 )

add_library(sg MODULE ${sg_SRCS})
add_dependencies(sg kame)
target_link_libraries(sg kame_sgcore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS sg LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
