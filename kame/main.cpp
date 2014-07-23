/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
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

#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QApplication>
#include <QMainWindow>

#include "kame.h"
#include "xsignal.h"
#include "icons/icon.h"
#include <QGLFormat>
#include <QFile>
#include <QTextCodec>
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

	QStringList module_dir;
	{
//		KGlobal::dirs()->addPrefix(".");
        makeIcons();
		{
            g_bLogDbgPrint = parser.isSet(logOption);
            
            g_bMLockAlways = parser.isSet(mlockAllOption);

			if(g_bMLockAlways) {
				if(( mlockall(MCL_CURRENT | MCL_FUTURE ) == 0)) {
					dbgPrint("MLOCKALL succeeded.");
				}
				else{
					dbgPrint(formatString("MLOCKALL failed errno=%d.", errno));
				}
			}

            g_bUseMLock = !parser.isSet(noMLockOption);
			if(g_bUseMLock)
                mlock(dummy_for_mlock, 4096uL); //reserve stack of main thread.
        
			QGLFormat f;
            f.setDirectRendering( !parser.isSet(noDirectRenderOption) );
			QGLFormat::setDefaultFormat( f );
            
            // Use UTF8 conversion from std::string to QString.
//            QTextCodec::setCodecForLocale(QTextCodec::codecForName("utf8") );
            
            module_dir = parser.values(moduleDirectoryOption);

#ifdef __SSE2__
			// Check CPU specs.
			if(cg_cpuSpec.verSSE < 2) {
				fprintf(stderr, "SSE2 is needed. Aborting.");
				return -1;
			}
#endif
			
			FrmKameMain *form;
			form = new FrmKameMain();
            
            if (args.count()) {
                form->openMes( args.at(0) );
			}
			else {
			}
            args.clear();
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
		QString path;
		path = *it + KAME_MODULE_DIR_SURFIX;
		lt_dladdsearchdir(path.toLocal8Bit().data());
		fprintf(stderr, "Searching for modules in %s\n", (const char*)path.toLocal8Bit().data());
		lt_dlforeachfile(path.toLocal8Bit().data(), &load_module, &modules);
	}

	for(auto it = modules.begin(); it != modules.end(); it++) {
		lt_dlhandle handle = lt_dlopenext(it->c_str());
		if(handle) {
			fprintf(stderr, "Module %s loaded\n", it->c_str());
		}
		else {
			fprintf(stderr, "Failed during loading module %s.\n", it->c_str());
		}
	}

    int ret = app.exec();

	return ret;
}  
