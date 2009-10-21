########### next target ###############
set(dmmcore_SRCS
    dmm.cpp 
 )

kde4_add_ui_files(dmmcore_SRCS
  dmmform.ui  )

kde4_add_library(dmmcore STATIC ${dmmcore_SRCS})
########### install files ###############
