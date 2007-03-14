pkglib_LTLIBRARIES = libdriver.la

INCLUDES = -I$(top_srcdir)/kame \
	$(all_includes)

libdriver_la_SOURCES = interface.cpp interface.h \
	dummydriver.h \
	dummyport.cpp dummyport.h \
	driver.cpp driver.h \
	primarydriver.cpp primarydriver.h \
	secondarydriver.cpp secondarydriver.h \
	charinterface.cpp charinterface.h \
	chardevicedriver.h \
	nidaqmxdriver.cpp nidaqmxdriver.h \
	oxforddriver.cpp oxforddriver.h \
	gpib.cpp gpib.h serial.cpp serial.h

METASOURCES = AUTO
