include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface)
    
########### next target ###############
set(counter_SRCS
    counter.cpp 
 )

kde4_add_library(counter MODULE ${lia_SRCS})
target_link_libraries(counter kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS counter LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

