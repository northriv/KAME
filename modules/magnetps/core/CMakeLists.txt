set(magnetpscore_SRCS
    magnetps.cpp
 )

kde4_add_ui_files(magnetpscore_SRCS
  magnetpsform.ui  )

kde4_add_library(kame_magnetpscore SHARED ${magnetpscore_SRCS})
########### install files ###############
install(TARGETS kame_magnetpscore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
