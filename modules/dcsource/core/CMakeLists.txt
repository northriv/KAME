########### next target ###############
set(dcsourcecore_SRCS
    dcsource.cpp 
 )

kde4_add_ui_files(dcsourcecore_SRCS
  dcsourceform.ui  )

kde4_add_library(dcsourcecore MODULE ${dcsourcecore_SRCS})
########### install files ###############
install(TARGETS dcsourcecore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
