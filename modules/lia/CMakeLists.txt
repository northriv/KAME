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
add_dependencies(lia kame)
########### install files ###############
install(TARGETS lia LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

