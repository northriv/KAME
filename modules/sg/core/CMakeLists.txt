########### next target ###############
set(sgcore_SRCS
	signalgenerator.cpp 
 )

kde4_add_ui_files(sgcore_SRCS
  signalgeneratorform.ui)
  
kde4_add_library(kame_sgcore SHARED ${sgcore_SRCS})
########### install files ###############
install(TARGETS kame_sgcore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
