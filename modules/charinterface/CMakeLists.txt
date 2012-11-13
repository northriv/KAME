include_directories(
    ${GPIB_INCLUDE_DIR})

########### next target ###############
set(charinterface_SRCS
	charinterface.cpp  
	dummyport.cpp 
	gpib.cpp  
	serial.cpp 
	tcp.cpp
	oxforddriver.cpp 
 )

add_library(kame_charinterface SHARED ${charinterface_SRCS})
target_link_libraries(kame_charinterface ${GPIB_LIBRARY})
########### install files ###############
install(TARGETS kame_charinterface LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
