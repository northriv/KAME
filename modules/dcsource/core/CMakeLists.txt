########### next target ###############
set(dcsourcecore_SRCS
    dcsource.cpp 
 )

kde4_add_ui_files(dcsourcecore_SRCS
  dcsourceform.ui  )

kde4_add_library(kame_dcsourcecore SHARED ${dcsourcecore_SRCS})
########### install files ###############
install(TARGETS kame_dcsourcecore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
