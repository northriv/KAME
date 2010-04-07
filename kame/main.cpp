/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "support.h"

#include <kcmdlineargs.h>
#include <kaboutdata.h>
#include <kapplication.h>

#include "kame.h"
#include "xsignal.h"
#include "icons/icon.h"
#include <kiconloader.h>
#include <kstandarddirs.h>
#include <QGLFormat>
#include <QFile>
#include <QTextCodec>
#include <errno.h>

#include <ltdl.h>

int load_module(const char *filename, lt_ptr data) {
	static_cast<std::deque<std::string> *>(data)->push_back(filename);
	return 0;
}

int main(int argc, char *argv[])
{
#ifdef HAVE_LIBGCCPP
	//initialize GC
	GC_INIT();
	// GC_find_leak = 1;
	//GC_dont_gc
#endif  
	const char *description =
	I18N_NOOP("KAME");
	// INSERT A DESCRIPTION FOR YOUR APPLICATION HERE
   
	KAboutData aboutData( "kame", "", ki18n("KAME"),
						  VERSION, ki18n(description), KAboutData::License_GPL,
						  ki18n("(c) 2003-2009"), ki18n(""), "", "kitag@issp.u-tokyo.ac.jp");
	KCmdLineArgs::init( argc, argv, &aboutData );

	KCmdLineOptions options;
	options.add("logging", ki18n("log debugging info."));
	options.add("mlockall", ki18n("never cause swapping, perhaps you need 'ulimit -l <MB>'"));
	options.add("nomlock", ki18n("never use mlock"));
	options.add("nodr");
	options.add("nodirectrender", ki18n("do not use direct rendering"));
	options.add("moduledir <path>", ki18n("search modules in <path> instead of the standard dirs"));
	options.add("+[File]", ki18n("measurement file to open"));

	KCmdLineArgs::addCmdLineOptions( options ); // Add our own options.

	KApplication *app;
	app = new KApplication;
  
	QStringList module_dir;
	{
		KGlobal::dirs()->addPrefix(".");
		makeIcons( KIconLoader::global());
		{
			KCmdLineArgs *args = KCmdLineArgs::parsedArgs();
                    
			g_bLogDbgPrint = args->isSet("logging");
            
			g_bMLockAlways = args->isSet("mlockall");

			if(g_bMLockAlways) {
				if(( mlockall(MCL_CURRENT | MCL_FUTURE ) == 0)) {
					dbgPrint("MLOCKALL succeeded.");
				}
				else{
					dbgPrint(formatString("MLOCKALL failed errno=%d.", errno));
				}
			}

			g_bUseMLock = args->isSet("mlock");
			if(g_bUseMLock)
				mlock(&aboutData, 4096uL); //reserve stack of main thread.
        
			QGLFormat f;
			f.setDirectRendering(args->isSet("directrender") );
			QGLFormat::setDefaultFormat( f );
            
			// Use UTF8 conversion from std::string to QString.
			QTextCodec::setCodecForCStrings(QTextCodec::codecForName("utf8") );
            
			module_dir = args->getOptionList("moduledir");

#ifdef __SSE2__
			// Check CPU specs.
			if(cg_cpuSpec.verSSE < 2) {
				fprintf(stderr, "SSE2 is needed. Aborting.");
				return -1;
			}
#endif
			
			FrmKameMain *form;
			form = new FrmKameMain();
            
			if (args->count())
			{
				form->openMes( args->arg(0) );
			}
			else
			{
			}
			args->clear();
		}
	}
	/*
	  while(!app->closingDown()) {
	  bool idle = g_signalBuffer->synchronize();  
	  if(idle) app->processEvents();
	  app->processEvents(15);
	  }
	  return 0;
	*/
	fprintf(stderr, "Start processing events.\n");

	app->processEvents();

	fprintf(stderr, "Initializing LTDL.\n");
	lt_dlinit();
#ifdef __linux__
	LTDL_SET_PRELOADED_SYMBOLS();
#endif
	if(module_dir.isEmpty())
		module_dir = KGlobal::dirs()->resourceDirs("lib");
	std::deque<XString> modules_core, modules;
	for(QStringList::iterator it = module_dir.begin(); it != module_dir.end(); it++) {
		QString path;
		path = *it + "kame/core_modules";
		lt_dladdsearchdir(path.toLocal8Bit().data());
		fprintf(stderr, "searching for core libraries in %s\n", (const char*)path.toLocal8Bit().data());
		lt_dlforeachfile(path.toLocal8Bit().data(), &load_module, &modules_core);
		path = *it + "kame/modules";
		lt_dladdsearchdir(path.toLocal8Bit().data());
		fprintf(stderr, "searching for libraries in %s\n", (const char*)path.toLocal8Bit().data());
		lt_dlforeachfile(path.toLocal8Bit().data(), &load_module, &modules);
	}

	modules_core.insert(modules_core.end(), modules.begin(), modules.end());
	for(std::deque<XString>::iterator it = modules_core.begin(); it != modules_core.end(); it++) {
		lt_dlhandle handle = lt_dlopenext(it->c_str());
		if(handle) {
			fprintf(stderr, "Module %s loaded\n", it->c_str());
		}
		else {
			fprintf(stderr, "loading module %s failed %s.\n", it->c_str(), lt_dlerror());
		}
	}

	int ret = app->exec();

	return ret;
}  
