INCLUDES = -I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/graph \
	-I$(top_builddir)/kame/graph $(all_includes)

METASOURCES = AUTO

noinst_LIBRARIES =  libforms.a
libforms_a_SOURCES = recordreaderform.ui \
	graphtool.ui \
	scalarentrytool.ui \
	drivertool.ui \
	interfacetool.ui \
	drivercreate.ui \
	rubythreadtool.ui \
	recordreaderconnector.cpp recordreaderconnector.h \
	driverlistconnector.cpp driverlistconnector.h \
	interfacelistconnector.cpp interfacelistconnector.h \
	graphlistconnector.cpp graphlistconnector.h \
	entrylistconnector.cpp entrylistconnector.h

	