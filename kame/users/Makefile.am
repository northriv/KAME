noinst_LIBRARIES = libusers.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/driver \
	$(all_includes)

libusers_a_SOURCES = testdriver.cpp testdriver.h \
	userdrivers.cpp 

SUBDIRS = dmm dso tempcontrol magnetps lia dcsource funcsynth nmr montecarlo

METASOURCES = AUTO

