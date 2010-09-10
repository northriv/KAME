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
########### install files ###############
install(TARGETS networkanalyzer LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)
