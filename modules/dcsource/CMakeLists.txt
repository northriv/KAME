add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/dcsource/core)

########### next target ###############
set(dcsource_SRCS
    userdcsource.cpp 
 )

add_library(dcsource MODULE ${dcsource_SRCS})
target_link_libraries(dcsource dcsourcecore charinterface)
########### install files ###############
install(TARGETS dcsource LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

