include_directories(
     ${CMAKE_SOURCE_DIR}/kame/math
     ${CMAKE_SOURCE_DIR}/kame/graph)

########### next target ###############
set(nmrpulsercore_SRCS
	pulserdriverconnector.cpp 
	pulserdriver.cpp 
 )

kde4_add_ui_files(nmrpulsercore_SRCS
  pulserdriverform.ui
  pulserdrivermoreform.ui  )

kde4_add_library(nmrpulsercore SHARED ${nmrpulsercore_SRCS})
########### install files ###############
install(TARGETS nmrpulsercore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
