########### next target ###############
set(levelmetercore_SRCS
    levelmeter.cpp 
 )
add_library(levelmetercore MODULE ${levelmetercore_SRCS})
########### install files ###############
install(TARGETS levelmetercore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules/core)
