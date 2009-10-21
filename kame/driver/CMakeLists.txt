########### next target ###############
set(driver_SRCS
    interface.cpp
	driver.cpp
	primarydriver.cpp
	secondarydriver.cpp)

kde4_add_library(driver STATIC ${driver_SRCS})
########### install files ###############
