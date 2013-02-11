set(motorcore_SRCS
    motor.cpp 
 )

kde4_add_ui_files(motorcore_SRCS
  motorform.ui  
 )
   
kde4_add_library(kame_motorcore SHARED ${motorcore_SRCS})
target_link_libraries(kame_motorcore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS kame_motorcore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})

