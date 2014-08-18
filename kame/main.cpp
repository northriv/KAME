/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "support.h"

#ifdef WITH_KDE
	#include <kcmdlineargs.h>
	#include <kaboutdata.h>
	#include <kapplication.h>
	#include <kstandarddirs.h>
#else
	#include <QCommandLineParser>
	#include <QCommandLineOption>
	#include <QApplication>
    #include <QMainWindow>
#endif

#include "kame.h"
#include "xsignal.h"
#include "icons/icon.h"
#include "messagebox.h"
#include <QGLFormat>
#include <QFile>
#include <QTextCodec>
#include <QTranslator>
#include <QLibraryInfo>
#ifndef WITH_KDE
    #include <QStandardPaths>
#endif
#include <errno.h>

#include <ltdl.h>

#include <gsl/gsl_errno.h>

void
my_gsl_err_handler (const char *reason, const char *file, int line, int gsl_errno) {
	gErrPrint_redirected(formatString("GSL emitted an error for a reason:%s; %s", reason, gsl_strerror(gsl_errno)), file, line);
}

int load_module(const char *filename, lt_ptr data) {
	static_cast<std::deque<std::string> *>(data)->push_back(filename);
	return 0;
}

int main(int argc, char *argv[]) {
    char dummy_for_mlock[2048];
#ifdef HAVE_LIBGCCPP
	//initialize GC
	GC_INIT();
	// GC_find_leak = 1;
	//GC_dont_gc
#endif  

	Q_INIT_RESOURCE(kame);

#ifdef WITH_KDE
	const char *description =
	I18N_NOOP("KAME");
	// INSERT A DESCRIPTION FOR YOUR APPLICATION HERE

	KAboutData aboutData( "kame", "", ki18n("KAME"),
						  VERSION, ki18n(description), KAboutData::License_GPL,
						  ki18n("(c) 2003-2009"), ki18n(""), "", "kitag@kochi-u.ac.jp");
	KCmdLineArgs::init( argc, argv, &aboutData );

	KCmdLineOptions options;
	options.add("logging", ki18n("log debugging info."));
	options.add("mlockall", ki18n("never cause swapping, perhaps you need 'ulimit -l <MB>'"));
	options.add("nomlock", ki18n("never use mlock"));
    options.add("overpaint", ki18n("draws text over OpenGL rendered scenes"));
    options.add("nodr");
	options.add("nodirectrender", ki18n("do not use direct rendering"));
	options.add("moduledir <path>", ki18n("search modules in <path> instead of the standard dirs"));
	options.add("+[File]", ki18n("measurement file to open"));

	KCmdLineArgs::addCmdLineOptions( options ); // Add our own options.

	KApplication app;

	KGlobal::dirs()->addPrefix(".");

	KCmdLineArgs *args = KCmdLineArgs::parsedArgs();
	g_bLogDbgPrint = args->isSet("logging");
    g_bUseOverpaint = args->isSet("overpaint");
    g_bMLockAlways = args->isSet("mlockall");
	g_bUseMLock = args->isSet("mlock");
	QStringList  module_dir = args->getOptionList("moduledir");
	if(module_dir.isEmpty())
		module_dir = KGlobal::dirs()->resourceDirs("lib");
    bool usedirectrender =args->isSet("directrender") ;

    XString mesfile = args->count() ? args->arg(0) : "";
    args->clear();
#else
    QApplication app(argc, argv);
    QApplication::setApplicationName("kame");
    QApplication::setApplicationVersion(VERSION);

    QCommandLineParser parser;
    parser.setApplicationDescription("KAME");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addPositionalArgument("file", QCoreApplication::translate("main", "Measurement file to open"));

    QCommandLineOption logOption(QStringList() << "l" << "logging", "Log debugging info.");
    parser.addOption(logOption);
    QCommandLineOption overpaintOption(QStringList() << "o" << "overpaint", "Draws text over OpenGL rendered scenes.");
    parser.addOption(overpaintOption);
    QCommandLineOption mlockAllOption(QStringList() << "m" << "mlockall",
          "Never cause swapping, perhaps you need 'ulimit -l <MB>'");
    parser.addOption(mlockAllOption);
    QCommandLineOption noMLockOption(QStringList() << "n" << "nomlock", "Never use mlock");
    parser.addOption(noMLockOption);
    QCommandLineOption noDirectRenderOption("nodr", QCoreApplication::translate("main", "Do not use direct rendering"));
    parser.addOption(noDirectRenderOption);

    QCommandLineOption moduleDirectoryOption("moduledir",
            QCoreApplication::translate("main", "search modules in <path> instead of the standard dirs"),
            QCoreApplication::translate("main", "path"));
    parser.addOption(moduleDirectoryOption);

    parser.process(app); //processes args.

    QStringList args = parser.positionalArguments();

    g_bLogDbgPrint = parser.isSet(logOption);
    g_bUseOverpaint = parser.isSet(overpaintOption);
    g_bMLockAlways = parser.isSet(mlockAllOption);
    g_bUseMLock = !parser.isSet(noMLockOption);
	QStringList  module_dir = parser.values(moduleDirectoryOption);
    bool usedirectrender = !parser.isSet(noDirectRenderOption);

    XString mesfile = args.count() ? args.at(0) : "";
    args.clear();


    QTranslator qtTranslator;
    qtTranslator.load("qt_" + QLocale::system().name(), QLibraryInfo::location(QLibraryInfo::TranslationsPath));
    app.installTranslator(&qtTranslator);

    QTranslator appTranslator;
    if( !appTranslator.load("kame_" + QLocale::system().name())) {
        appTranslator.load("kame_" + QLocale::system().name(), app.applicationDirPath());
    }
    app.installTranslator(&appTranslator);
#endif

//#if defined __WIN32__ || defined WINDOWS
//    if(AllocConsole()) {
//        freopen("CONOUT$", "w", stdout);
//        freopen("CONOUT$", "w", stderr);
//    }
//#endif

	{
        makeIcons();
		{

#if !defined __WIN32__ && !defined WINDOWS
			if(g_bMLockAlways) {
				if(( mlockall(MCL_CURRENT | MCL_FUTURE ) == 0)) {
					dbgPrint("MLOCKALL succeeded.");
				}
				else{
					dbgPrint(formatString("MLOCKALL failed errno=%d.", errno));
				}
			}

			if(g_bUseMLock)
                mlock(dummy_for_mlock, 4096uL); //reserve stack of main thread.
#endif

            if( !usedirectrender) {
                QGLFormat f;
                f.setDirectRendering( false);
                QGLFormat::setDefaultFormat( f );
            }
            
            // Use UTF8 conversion from std::string to QString.
//            QTextCodec::setCodecForLocale(QTextCodec::codecForName("utf8") );
            
#ifdef __SSE2__
			// Check CPU specs.
			if(cg_cpuSpec.verSSE < 2) {
				fprintf(stderr, "SSE2 is needed. Aborting.");
				return -1;
			}
#endif
			
			FrmKameMain *form;
			form = new FrmKameMain();
            
            if(mesfile.length()) {
                form->openMes(mesfile);
            }
		}
	}

	//Overrides GSL's error handler.
	gsl_set_error_handler(&my_gsl_err_handler);

	/*
	  while(!app->closingDown()) {
	  bool idle = g_signalBuffer->synchronize();  
	  if(idle) app->processEvents();
	  app->processEvents(15);
	  }
	  return 0;
	*/
	fprintf(stderr, "Start processing events.\n");

    app.processEvents();

	fprintf(stderr, "Initializing LTDL.\n");
	lt_dlinit();
#ifdef __linux__
	LTDL_SET_PRELOADED_SYMBOLS();
#endif
	if(module_dir.isEmpty())
        module_dir = app.libraryPaths();
    std::deque<XString> modules;
	for(auto it = module_dir.begin(); it != module_dir.end(); it++) {
        QStringList paths;
#if defined KAME_COREMODULE_DIR_SURFIX
        paths += *it + KAME_COREMODULE_DIR_SURFIX;
#endif
#if defined KAME_COREMODULE2_DIR_SURFIX
        paths += *it + KAME_COREMODULE2_DIR_SURFIX;
#endif
        paths += *it + KAME_MODULE_DIR_SURFIX;

        for(auto sit = paths.begin(); sit != paths.end(); sit++) {
            lt_dladdsearchdir(sit->toLocal8Bit().data());
            XMessageBox::post("Searching for modules in " + *sit, *g_pIconInfo);
            lt_dlforeachfile(sit->toLocal8Bit().data(), &load_module, &modules);
        }
    }

	for(auto it = modules.begin(); it != modules.end(); it++) {
		lt_dlhandle handle = lt_dlopenext(it->c_str());
		if(handle) {
            XMessageBox::post("Module \"" + *it + "\" loaded", *g_pIconKame);
		}
		else {
            XMessageBox::post("Failure during loading module \"" + *it + "\"", *g_pIconError);
		}
	}

    int ret = app.exec();

//#if defined __WIN32__ || defined WINDOWS
//    FreeConsole();
//#endif

	return ret;
}  
