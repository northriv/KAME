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

set(script_HEADERS
	xdotwriter.h
	xrubysupport.h
	xrubythread.h
	xrubythreadconnector.h
	xrubywriter.h
	)
	
QT4_WRAP_CPP(fscript_MOC ${script_HEADERS})

add_library(script STATIC ${script_MOC} ${script_SRCS})
add_dependencies(script forms)

########### install files ###############

install(FILES  rubylineshell.rb DESTINATION ${DATA_INSTALL_DIR}/kame)
