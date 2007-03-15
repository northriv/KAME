bin_PROGRAMS = kame

# set the include path for X, qt and KDE
INCLUDES = -I$(top_srcdir)/. \
	-I$(top_srcdir)/kame \
	-I$(top_srcdir)/kame/forms \
	-I$(top_srcdir)/kame/script \
	-I$(top_srcdir)/kame/driver \
	-I$(top_srcdir)/kame/analyzer \
	-I$(top_srcdir)/kame/thermometer \
	-I$(top_builddir)/kame/forms \
	-I$(top_builddir)/kame/script/forms \
	-I$(top_builddir)/kame/thermometer/forms \
	-I$(top_srcdir)/kame/icons \
	$(LTDLINCL) \
	$(all_includes)
	
kame_SOURCES = measure.cpp measure.h \
	kame.cpp kame.h \
	xnodeconnector.cpp xnodeconnector.h \
	xsignal.cpp xsignal.h xsignal_prv.h \
	xscheduler.cpp xscheduler.h \
	thread.cpp thread.h \
	atomic.h atomic_prv_x86.h atomic_prv_ppc.h \
	atomic_smart_ptr.h atomic_queue.h \
	threadlocal.cpp threadlocal.h \
	rwlock.cpp rwlock.h \
	support.cpp support.h \
	xtime.cpp xtime.h \
	xnode.cpp xnode.h \
	xlistnode.cpp xlistnode.h \
	xitemnode.cpp xitemnode.h \
	main.cpp
kame_LDADD = $(top_builddir)/kame/forms/libforms.a \
	$(top_builddir)/kame/script/libscript.a \
	$(top_builddir)/kame/script/forms/libforms.a \
	$(top_builddir)/kame/analyzer/libanalyzer.a \
	$(top_builddir)/kame/driver/libdriver.la \
	$(top_builddir)/kame/thermometer/libthermometer.a \
	$(top_builddir)/kame/thermometer/forms/libforms.a \
	$(top_builddir)/kame/icons/libicons.a \
	$(top_builddir)/kame/graph/libgraph.a \
	$(LIBLTDL) \
	$(top_srcdir)/FTGL/unix/src/.libs/libftgl.a \
	-lGLU -lm \
	$(LIB_KFILE) $(LIB_KDEUI) $(LIB_KDECORE) $(LIB_QT) $(LIBSOCKET) -lkmdi @RUBY_LIBS@\
	@GC_LIBS@ @MALLOC_DEBUG_LIBS@ @FREETYPE2_LIBS@ @GSL_LIBS@

kame_LDFLAGS = $(all_libraries) -export-dynamic
#-rdynamic
# $(KDE_RPATH)

SUBDIRS = forms script driver graph thermometer analyzer icons mikachanfont

EXTRA_DIST = kame.desktop lo32-app-kame.png lo16-app-kame.png

install-data-local:
	$(mkinstalldirs) $(kde_appsdir)/Applications/
	$(INSTALL_DATA) $(srcdir)/kame.desktop $(kde_appsdir)/Applications/kame.desktop
	$(mkinstalldirs) $(kde_icondir)/hicolor/32x32/apps/
	$(INSTALL_DATA) $(srcdir)/hi32-app-kame.png $(kde_icondir)/hicolor/32x32/apps/kame.png
	$(mkinstalldirs) $(kde_icondir)/hicolor/16x16/apps/
	$(INSTALL_DATA) $(srcdir)/hi16-app-kame.png $(kde_icondir)/hicolor/16x16/apps/kame.png

uninstall-local:
	-rm -f $(kde_appsdir)/Applications/kame.desktop
	-rm -f $(kde_icondir)/hicolor/32x32/apps/kame.png
	-rm -f $(kde_icondir)/hicolor/16x16/apps/kame.png

# These paths are KDE specific. Use them:
# kde_appsdir         Where your application's menu entry (.desktop) should go to.
# kde_icondir         Where your icon should go to - better use KDE_ICON.
# kde_sounddir        Where your sounds should go to.
# kde_htmldir         Where your docs should go to. (contains lang subdirs)
# kde_datadir         Where you install application data. (Use a subdir)
# kde_locale          Where translation files should go to. (contains lang subdirs)
# kde_cgidir          Where cgi-bin executables should go to.
# kde_confdir         Where config files should go to (system-wide ones with default values).
# kde_mimedir         Where mimetypes .desktop files should go to.
# kde_servicesdir     Where services .desktop files should go to.
# kde_servicetypesdir Where servicetypes .desktop files should go to.
# kde_toolbardir      Where general toolbar icons should go to (deprecated, use KDE_ICON).
# kde_wallpaperdir    Where general wallpapers should go to.
# kde_templatesdir    Where templates for the "New" menu (Konqueror/KDesktop) should go to.
# kde_bindir          Where executables should go to. Use bin_PROGRAMS or bin_SCRIPTS.
# kde_libdir          Where shared libraries should go to. Use lib_LTLIBRARIES.
# kde_moduledir       Where modules (e.g. parts) should go to. Use kde_module_LTLIBRARIES.
# kde_styledir        Where Qt/KDE widget styles should go to (new in KDE 3).
# kde_designerdir     Where Qt Designer plugins should go to (new in KDE 3).

METASOURCES = AUTO

rcdir = $(kde_datadir)/kame
rc_DATA = 

messages: rc.cpp
	LIST=`find . -name \*.h -o -name \*.hh -o -name \*.H -o -name \*.hxx -o -name \*.hpp -o -name \*.cpp -o -name \*.cc -o -name \*.cxx -o -name \*.ecpp -o -name \*.C`; \
	if test -n "$$LIST"; then \
	 $(XGETTEXT) -ki18n -kI18N_NOOP -ktr2i18n $$LIST -o ../po/kame.pot; \
	fi
