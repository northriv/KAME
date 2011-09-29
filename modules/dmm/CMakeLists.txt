add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/dmm/core)

########### next target ###############
set(dmm_SRCS
    userdmm.cpp 
 )

add_library(dmm MODULE ${dmm_SRCS})
target_link_libraries(dmm kame_dmmcore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS dmm LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
