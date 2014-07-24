########### next target ###############
set(driver_SRCS
    interface.cpp
	driver.cpp
	primarydriver.cpp
	secondarydriver.cpp)

set(driver_HEADERS
	driver.h
	dummydriver.h
	interface.h
	primarydriver.h
	primarydriverwiththread.h
	secondarydriver.h
	secondarydriverinterface.h
	)
	
QT4_WRAP_CPP(driver_MOC ${driver_HEADERS})

add_library(driver STATIC ${driver_SRCS} ${driver_MOC})
########### install files ###############
