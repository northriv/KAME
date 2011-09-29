include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/kame/graph )
    
########### next target ###############
set(networkanalyzer_SRCS
    networkanalyzer.cpp
    usernetworkanalyzer.cpp
 )

kde4_add_ui_files(networkanalyzer_SRCS
  networkanalyzerform.ui  )

kde4_add_library(networkanalyzer MODULE ${networkanalyzer_SRCS})
add_dependencies(networkanalyzer kame)
target_link_libraries(networkanalyzer kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS networkanalyzer LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
