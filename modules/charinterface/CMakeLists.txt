include_directories(
    ${LINUXGPIB_INCLUDE_DIR})

########### next target ###############
set(charinterface_SRCS
	charinterface.cpp  
	dummyport.cpp 
	gpib.cpp  
	serial.cpp 
	oxforddriver.cpp 
 )

add_library(charinterface MODULE ${charinterface_SRCS})
target_link_libraries(charinterface ${LINUXGPIB_LIBRARY})
########### install files ###############
install(TARGETS charinterface LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
