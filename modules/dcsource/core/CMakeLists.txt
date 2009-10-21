########### next target ###############
set(dcsourcecore_SRCS
    dcsource.cpp 
 )

kde4_add_ui_files(dcsourcecore_SRCS
  dcsourceform.ui  )

kde4_add_library(dcsourcecore STATIC ${dcsourcecore_SRCS})
########### install files ###############
