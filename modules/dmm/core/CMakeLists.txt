########### next target ###############
set(dmmcore_SRCS
    dmm.cpp 
 )

kde4_add_ui_files(dmmcore_SRCS
  dmmform.ui  )

kde4_add_library(kame_dmmcore SHARED ${dmmcore_SRCS})
########### install files ###############
install(TARGETS kame_dmmcore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
