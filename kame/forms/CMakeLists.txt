include_directories(
    ${CMAKE_SOURCE_DIR}/kame/driver
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/graph
    ${QT_INCLUDES} )

########### next target ###############

set(forms_SRCS
	recordreaderconnector.cpp
	driverlistconnector.cpp
	interfacelistconnector.cpp
	graphlistconnector.cpp
	entrylistconnector.cpp
	nodebrowser.cpp
 )

set(forms_FORMS
    recordreaderform.ui
	graphtool.ui 
	scalarentrytool.ui 
	drivertool.ui 
	interfacetool.ui 
    drivercreate.ui 
    rubythreadtool.ui 
    nodebrowserform.ui
    caltableform.ui)

set(forms_HEADERS
	driverlistconnector.h
	entrylistconnector.h
	graphlistconnector.h
	interfacelistconnector.h
	nodebrowser.h
	recordreaderconnector.h
	)
	
QT4_WRAP_UI(forms_FORMS_HEADERS ${forms_FORMS})
QT4_WRAP_CPP(forms_MOC ${forms_HEADERS})

add_library(forms STATIC ${forms_MOC} ${forms_FORMS_HEADERS} ${forms_SRCS})

########### install files ###############

