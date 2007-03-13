noinst_LIBRARIES = libthermometer.a

INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph \
	-I$(top_srcdir)/kame/thermometer \
	-I$(top_builddir)/kame/thermometer/forms \
	$(all_includes)

libthermometer_a_SOURCES = \
	caltable.cpp caltable.h \
	thermometer.cpp thermometer.h \
	cspline.cpp cspline.h

SUBDIRS = forms
METASOURCES = AUTO
