########### next target ###############
set(sgcore_SRCS
	signalgenerator.cpp 
 )

kde4_add_ui_files(sgcore_SRCS
  signalgeneratorform.ui)
  
kde4_add_library(sgcore SHARED ${sgcore_SRCS})
########### install files ###############
install(TARGETS sgcore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
