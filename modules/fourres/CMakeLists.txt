include_directories(
    ${CMAKE_SOURCE_DIR}/modules/dmm/core
    ${CMAKE_SOURCE_DIR}/modules/dcsource/core)
########### next target ###############
set(fourres_SRCS
	fourres.cpp 
 )

kde4_add_ui_files(fourres_SRCS
  fourresform.ui)
  
kde4_add_library(fourres MODULE ${fourres_SRCS})
target_link_libraries(fourres kame_dcsourcecore kame_dmmcore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS fourres LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

