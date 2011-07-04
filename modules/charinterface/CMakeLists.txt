include_directories(
    ${GPIB_INCLUDE_DIR})

########### next target ###############
set(charinterface_SRCS
	charinterface.cpp  
	dummyport.cpp 
	gpib.cpp  
	serial.cpp 
	oxforddriver.cpp 
 )

add_library(charinterface SHARED ${charinterface_SRCS})
target_link_libraries(charinterface ${GPIB_LIBRARY})
add_dependencies(charinterface kame)
########### install files ###############
install(TARGETS charinterface LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/core_modules)
