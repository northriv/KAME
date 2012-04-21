include_directories(
    ${CMAKE_SOURCE_DIR}/kame/graph )
    
########### next target ###############
set(networkanalyzercore_SRCS
    networkanalyzer.cpp
 )

kde4_add_ui_files(networkanalyzercore_SRCS
  networkanalyzerform.ui  )

kde4_add_library(kame_networkanalyzercore SHARED ${networkanalyzercore_SRCS})
########### install files ###############
install(TARGETS kame_networkanalyzercore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
