noinst_LIBRARIES = libdriver.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_builddir)/kame/forms \
	-I$(top_builddir)/kame/driver/forms \
	$(all_includes)

libdriver_a_SOURCES = interface.cpp interface.h \
	charinterface.cpp charinterface.h \
	chardevicedriver.h \
	nidaqmxdriver.cpp nidaqmxdriver.h \
	dummydriver.h \
	oxforddriver.cpp oxforddriver.h \
	gpib.cpp gpib.h serial.cpp serial.h \
	dummyport.cpp dummyport.h \
	driver.cpp driver.h \
	primarydriver.cpp primarydriver.h \
	secondarydriver.cpp secondarydriver.h \
	driverlistconnector.cpp driverlistconnector.h \
	interfacelistconnector.cpp interfacelistconnector.h 

METASOURCES = AUTO

SUBDIRS = forms
