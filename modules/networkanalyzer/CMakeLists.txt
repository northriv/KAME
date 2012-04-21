add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/networkanalyzer/core )
    
########### next target ###############
set(networkanalyzer_SRCS
    usernetworkanalyzer.cpp
 )

kde4_add_library(networkanalyzer MODULE ${networkanalyzer_SRCS})
target_link_libraries(networkanalyzer kame_networkanalyzercore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS networkanalyzer LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
