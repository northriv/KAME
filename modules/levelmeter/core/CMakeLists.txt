########### next target ###############
set(levelmetercore_SRCS
    levelmeter.cpp 
 )
add_library(levelmetercore SHARED ${levelmetercore_SRCS})
########### install files ###############
install(TARGETS levelmetercore LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
