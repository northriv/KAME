########### next target ###############
set(dmmcore_SRCS
    dmm.cpp 
 )

kde4_add_ui_files(dmmcore_SRCS
  dmmform.ui  )

kde4_add_library(dmmcore MODULE ${dmmcore_SRCS})
########### install files ###############
install(TARGETS dmmcore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules/core)
