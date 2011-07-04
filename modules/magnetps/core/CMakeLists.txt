set(magnetpscore_SRCS
    magnetps.cpp
 )

kde4_add_ui_files(magnetpscore_SRCS
  magnetpsform.ui  )

kde4_add_library(magnetpscore SHARED ${magnetpscore_SRCS})
########### install files ###############
install(TARGETS magnetpscore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
