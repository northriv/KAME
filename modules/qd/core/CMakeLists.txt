include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface)
    
########### next target ###############
set(lia_SRCS
    lockinamp.cpp 
    userlockinamp.cpp 
 )

kde4_add_ui_files(lia_SRCS
  lockinampform.ui  )

kde4_add_library(lia MODULE ${lia_SRCS})
target_link_libraries(lia kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS lia LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

