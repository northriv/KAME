include_directories(
    )

set(flowcontrollercore_SRCS
    flowcontroller.cpp 
 )

kde4_add_ui_files(flowcontrollercore_SRCS
  flowcontrollerform.ui  
  )
   
kde4_add_library(kame_flowcontrollercore SHARED ${flowcontrollercore_SRCS})
target_link_libraries(kame_flowcontrollercore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS kame_flowcontrollerrcore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})

