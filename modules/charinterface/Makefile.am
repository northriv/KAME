pkglib_LTLIBRARIES = libcharinterface.la

libcharinterface_la_LDFLAGS = 

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	$(all_includes)
	
libcharinterface_la_SOURCES = \
	charinterface.cpp charinterface.h \
	chardevicedriver.h \
	dummyport.cpp dummyport.h \
	gpib.cpp gpib.h serial.cpp serial.h \
	oxforddriver.cpp oxforddriver.h

libcharinterface_la_LIBADD = @LINUXGPIB_LIBS@

METASOURCES = AUTO
