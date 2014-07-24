include_directories(
    )

set(flowcontrollercore_SRCS
    flowcontroller.cpp 
 )

kde4_add_ui_files(flowcontrollercore_SRCS
  flowcontrollerform.ui  
  )
   
kde4_add_library(kame_flowcontrollercore SHARED ${flowcontrollercore_SRCS})
########### install files ###############
install(TARGETS kame_flowcontrollercore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})

