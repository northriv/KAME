include_directories(
    ${CMAKE_SOURCE_DIR}/kame/math)

########### next target ###############
set(testdriver_SRCS
	testdriver.cpp 
 )

add_library(testdriver MODULE ${testdriver_SRCS})
target_link_libraries(testdriver ${MODULE_LINKER_FLAGS})
add_dependencies(testdriver kame)
########### install files ###############
install(TARGETS testdriver LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

