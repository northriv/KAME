pkglib_LTLIBRARIES = libdriver.la

INCLUDES = -I$(top_srcdir)/kame \
	$(all_includes)

libdriver_la_LDFLAGS = -export-dynamic

libdriver_la_SOURCES = interface.cpp interface.h \
	dummydriver.h \
	driver.cpp driver.h \
	primarydriver.cpp primarydriver.h \
	secondarydriver.cpp secondarydriver.h

METASOURCES = AUTO
