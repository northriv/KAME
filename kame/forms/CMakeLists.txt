include_directories(
    ${CMAKE_SOURCE_DIR}/kame/driver
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )

########### next target ###############

set(forms_SRCS
	recordreaderconnector.cpp
	driverlistconnector.cpp
	interfacelistconnector.cpp
	graphlistconnector.cpp
	entrylistconnector.cpp
	nodebrowser.cpp
 )

kde4_add_ui_files(forms_SRCS
    recordreaderform.ui
	graphtool.ui 
	scalarentrytool.ui 
	drivertool.ui 
	interfacetool.ui 
    drivercreate.ui 
    rubythreadtool.ui 
    nodebrowserform.ui
    caltableform.ui)

kde4_add_library(forms STATIC ${forms_SRCS})

########### install files ###############

