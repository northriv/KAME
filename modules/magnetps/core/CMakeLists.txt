set(magnetpscore_SRCS
    magnetps.cpp
 )

kde4_add_ui_files(magnetpscore_SRCS
  magnetpsform.ui  )

kde4_add_library(magnetpscore STATIC ${magnetpscore_SRCS})
