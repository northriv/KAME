include_directories(
    ${CMAKE_SOURCE_DIR}/modules/networkanalyzer/core
    )

set(motorcore_SRCS
    motor.cpp 
    autolctuner.cpp
 )

kde4_add_ui_files(motorcore_SRCS
  motorform.ui  
  autolctunerform.ui )
   
kde4_add_library(kame_motorcore networkanalyzercore SHARED ${motorcore_SRCS})
########### install files ###############
install(TARGETS kame_motorcore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})

