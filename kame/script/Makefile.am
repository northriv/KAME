noinst_LIBRARIES = libscript.a

INCLUDES = -I$(top_srcdir)/kame\
	-I$(top_builddir)/kame/forms\
	 $(all_includes)

libscript_a_SOURCES = xrubythread.cpp xrubythread.h \
	xrubythreadconnector.cpp xrubythreadconnector.h \
	xrubysupport.cpp xrubysupport.h \
	xrubywriter.cpp xrubywriter.h \
	xdotwriter.cpp xdotwriter.h 

AM_CPPFLAGS = 

METASOURCES = AUTO

rcdir = $(kde_datadir)/kame
rc_DATA = xrubysupport.rb rubyinterrupter.rb

