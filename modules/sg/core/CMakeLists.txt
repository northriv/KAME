########### next target ###############
set(sgcore_SRCS
	signalgenerator.cpp 
 )

kde4_add_ui_files(sgcore_SRCS
  signalgeneratorform.ui)
  
kde4_add_library(sgcore STATIC ${sgcore_SRCS})
########### install files ###############
