add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/dmm/core)

########### next target ###############
set(dmm_SRCS
    userdmm.cpp 
 )

add_library(dmm MODULE ${dmm_SRCS})
target_link_libraries(dmm dmmcore charinterface)
########### install files ###############
install(TARGETS dmm LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)
