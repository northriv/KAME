########### next target ###############
set(levelmetercore_SRCS
    levelmeter.cpp 
 )
add_library(kame_levelmetercore SHARED ${levelmetercore_SRCS})
########### install files ###############
install(TARGETS kame_levelmetercore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
