include_directories(
    ${CMAKE_BINARY_DIR}/kame/forms
    ${RUBY_INCLUDE_PATH} )

########### next target ###############

set(script_SRCS
	xrubythread.cpp
	xrubythreadconnector.cpp
	xrubysupport.cpp
	xrubywriter.cpp
	xdotwriter.cpp
 )

kde4_add_library(script STATIC ${script_SRCS})
########### install files ###############

install(FILES  xrubysupport.rb rubylineshell.rb DESTINATION ${DATA_INSTALL_DIR}/kame)
