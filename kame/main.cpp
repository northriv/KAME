/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "support.h"
#include <iostream>

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
#include "icons/icon.h"
#include "messagebox.h"
#include "allocator.h"  // KamePooledAllocGuard (kame/allocator.h shim → kamepoolalloc/allocator.h).  Was pulled transitively via kamestm/transaction_signal.h before the kamestm-from-kamepoolalloc decoupling.
                        // allocator.h also provides the §30 realtime-mode
                        // no-op stub on USE_STD_ALLOCATOR builds (MSVC),
                        // so we can call kame_pool_set_realtime_mode()
                        // unconditionally below without a #ifdef guard.
#ifndef USE_STD_ALLOCATOR
#  include "kame_pool.h"  // (§30) kame_pool_set_realtime_mode — extern "C" decl
#endif
#include <QFile>
#include <QTextCodec>
#include <QTranslator>
#include <QLibraryInfo>
#ifndef WITH_KDE
    #include <QStandardPaths>
#endif
#include <errno.h>

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #define NOMINMAX
    #include <windows.h>
    #include <QDir>
    #define USE_LOADLIBRARY
#else
    #define USE_LIBTOOL
    #include <ltdl.h>
#endif

#if defined __MACOSX__ || defined __APPLE__
    #include "support_osx.h"
#endif

#include <gsl/gsl_errno.h>

void
my_gsl_err_handler (const char *reason, const char *file, int line, int gsl_errno) {
//	gErrPrint_redirected(formatString("GSL emitted an error for a reason:%s; %s", reason, gsl_strerror(gsl_errno)), file, line);
    fprintf(stderr, "GSL emitted an error for a reason:%s; %s, at %s:%d\n", reason, gsl_strerror(gsl_errno), file, line);
}

#ifdef USE_LIBTOOL
int load_module(const char *filename, lt_ptr data) {
    static_cast<std::deque<XString> *>(data)->push_back(QString::fromLocal8Bit(filename));
	return 0;
}
//! Shared dlopen options for every module — see the lt_dladvise_global()
//! call in main() for why RTLD_GLOBAL is required rather than merely nice.
static lt_dladvise g_dl_advise;
#endif

int main(int argc, char *argv[]) {
    char dummy_for_mlock[8192];

	Q_INIT_RESOURCE(kame);

#ifdef WITH_KDE
	const char *description =
	I18N_NOOP("KAME");
	// INSERT A DESCRIPTION FOR YOUR APPLICATION HERE

	KAboutData aboutData( "kame", "", ki18n("KAME"),
						  VERSION, ki18n(description), KAboutData::License_GPL,
                          ki18n("(c) 2003-2014"), ki18n(""), "", "kitag@issp.u-tokyo.ac.jp");
	KCmdLineArgs::init( argc, argv, &aboutData );

	KCmdLineOptions options;
	options.add("logging", ki18n("log debugging info."));
	options.add("nomlock", ki18n("never use mlock"));
    options.add("nodr");
	options.add("moduledir <path>", ki18n("search modules in <path> instead of the standard dirs"));
	options.add("+[File]", ki18n("measurement file to open"));

	KCmdLineArgs::addCmdLineOptions( options ); // Add our own options.

	KApplication app;

	KGlobal::dirs()->addPrefix(".");

	KCmdLineArgs *args = KCmdLineArgs::parsedArgs();
	g_bLogDbgPrint = args->isSet("logging");
	g_bUseMLock = args->isSet("mlock");
	QStringList  module_dir = args->getOptionList("moduledir");
	if(module_dir.isEmpty())
		module_dir = KGlobal::dirs()->resourceDirs("lib");

    XString mesfile = args->count() ? args->arg(0) : "";
    args->clear();
#else
    QApplication app(argc, argv);
    QApplication::setApplicationName("kame");
    QApplication::setApplicationVersion(VERSION);
    app.setAttribute(Qt::AA_DontShowIconsInMenus, false); //In recent Mac/Qt, icons hidden by default.

    QCommandLineParser parser;
    parser.setApplicationDescription("KAME");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addPositionalArgument("file", QCoreApplication::translate("main", "Measurement file to open"));

    QCommandLineOption logOption(QStringList() << "l" << "logging", "Log debugging info.");
    parser.addOption(logOption);
    QCommandLineOption noMLockOption(QStringList() << "n" << "nomlock", "Never use mlock");
    parser.addOption(noMLockOption);

    QCommandLineOption moduleDirectoryOption("moduledir",
            QCoreApplication::translate("main", "search modules in <path> instead of the standard dirs"),
            QCoreApplication::translate("main", "path"));
    parser.addOption(moduleDirectoryOption);

    parser.process(app); //processes args.

    QStringList args = parser.positionalArguments();

    g_bLogDbgPrint = parser.isSet(logOption);
    g_bUseMLock = !parser.isSet(noMLockOption);
	QStringList  module_dir = parser.values(moduleDirectoryOption);

    XString mesfile = args.count() ? args.at(0) : "";
    args.clear();


    QTranslator qtTranslator;
    qtTranslator.load("qt_" + QLocale::system().name(), QLibraryInfo::location(QLibraryInfo::TranslationsPath));
    app.installTranslator(&qtTranslator); //transaltions for QT.

    QTranslator appTranslator;
    if( !appTranslator.load("kame_" + QLocale::system().name())) {
        appTranslator.load("kame_" + QLocale::system().name(), app.applicationDirPath());
    }
    app.installTranslator(&appTranslator); //translations for KAME.
#endif

//#if defined __WIN32__ || defined WINDOWS || defined _WIN32
//    if(AllocConsole()) {
//        freopen("CONOUT$", "w", stdout);
//        freopen("CONOUT$", "w", stderr);
//    }
//#endif

    FrmKameMain *form;
    {
        makeIcons(); //loads icon pixmaps.
		{

            if(isMemLockAvailable())
                mlock(dummy_for_mlock, sizeof(dummy_for_mlock)); //reserve stack of main thread.

            // Use UTF8 conversion from std::string to QString.
//            QTextCodec::setCodecForLocale(QTextCodec::codecForName("utf8") );
            
#ifdef __SSE2__
			// Check CPU specs.
			if(cg_cpuSpec.verSSE < 2) {
				fprintf(stderr, "SSE2 is needed. Aborting.");
				return -1;
            }
#endif
            Transactional::setCurrentPriorityMode(Priority::UI_DEFERRABLE);
//            Transactional::setCurrentPriorityMode(Priority::NORMAL);

            app.setStyleSheet(
                "QGroupBox {"
                "  border: 1px solid palette(mid);"
                "  border-radius: 4px;"
                "  margin-top: 8px;"
                "  padding-top: 4px;"
                "}"
                "QGroupBox::title {"
                "  subcontrol-origin: margin;"
                "  left: 6px;"
                "  padding: 0 2px;"
                "}");

			form = new FrmKameMain();

            if(mesfile.length()) {
                form->openMes(mesfile);
            }
		}
	}

	//Overrides GSL's error handler.
	gsl_set_error_handler(&my_gsl_err_handler);

    fprintf(stderr, "Start processing events.\n");

    app.processEvents(); //displays a main window.

#ifdef USE_LIBTOOL
    fprintf(stderr, "Initializing LTDL.\n");
    lt_dlinit();
    // NOTE: no LTDL_SET_PRELOADED_SYMBOLS() here.  That macro registers
    // modules linked STATICALLY into the executable, and it expands to a
    // reference to `lt__PROGRAM__LTX_preloaded_symbols`, a symbol only
    // libtool itself emits when it drives the link.  KAME's autotools build
    // did; the qmake build does not, so on Linux the call was an undefined
    // reference at link time.  Every KAME module is a real shared object
    // opened from disk by lt_dlopenext() below, so there is nothing to
    // preload and nothing is lost by leaving it out.

    // Open every module with RTLD_GLOBAL, so a module's symbols are visible
    // to the modules loaded AFTER it.  This is not an optimisation — it is
    // what the coremodules -> coremodules2 -> modules load order exists for:
    // the leaf drivers genuinely reference symbols defined in their core
    // module and in charinterface (e.g. libdmm needs 7 symbols from
    // libdmmcore and 5 from libcharinterface).  ltdl's default is
    // RTLD_LOCAL, under which those stay unresolved and every dependent
    // module fails to open — and ltdl reports that as the unhelpful "file
    // not found", so the whole leaf half of the driver set silently
    // disappears.  macOS gets the same effect today from `-undefined
    // dynamic_lookup` + flat lookup; stating it explicitly makes the two
    // platforms agree rather than leaving one of them accidental.
    lt_dladvise_init( &g_dl_advise);
    lt_dladvise_global( &g_dl_advise);
    lt_dladvise_ext( &g_dl_advise);       //!< try each platform's suffixes, like lt_dlopenext
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
        paths += *it + KAME_COREMODULE2_DIR_SURFIX; //modules that depend on core ones
#endif
        paths += *it + KAME_MODULE_DIR_SURFIX; //modules that depend on core/core2

        //searches module directories
        for(auto sit = paths.begin(); sit != paths.end(); sit++) {
#ifdef USE_LIBTOOL
            lt_dladdsearchdir(sit->toLocal8Bit().data());
#endif
            XMessageBox::post("Searching for modules in " + *sit, *g_pIconInfo);
#ifdef USE_LIBTOOL
            lt_dlforeachfile(sit->toLocal8Bit().data(), &load_module, &modules);
#endif
#ifdef USE_LOADLIBRARY
            QFileInfoList files = QDir(*sit).entryInfoList(QStringList("*.dll"), QDir::Files);
            for(QFileInfoList::const_iterator it = files.constBegin(); it != files.constEnd(); ++it) {
                modules.push_back(it->filePath());
            }
#endif
        }
    }

    //defers loading python modules.
    for(auto it = modules.begin(); it != modules.end();) {
        if(it->find("python") != std::string::npos) {
            auto f = *it;
            if(f == modules.back())
                break;
            it = modules.erase(it);
            modules.push_back(f);
        }
        else
            it++;
    }

    // Known-deprecated module substrings — installs sometimes leave
    // stale dylibs from removed/renamed modules in the modules
    // directory; loading them against a fresh kame binary causes
    // misleading crashes (mismatched ABI for STM internals etc.).
    // List entries are case-insensitive sub-strings of the file name.
    static const char *const deprecated_modules[] = {
        "fourres",
    };

    int num_loaded_modules = 0;
    //loads modules.
    //
    // MULTI-PASS.  A module may depend on symbols defined in ANOTHER module
    // (leaf drivers on their `*core` module and on charinterface; nidaq on
    // dsocore + nmrpulsercore), and vtables/typeinfo are DATA symbols, so
    // they must resolve at load time — a dependency loaded later is too
    // late.  The coremodules -> coremodules2 -> modules directory order
    // expresses the coarse layering, but not the order WITHIN a directory,
    // which is whatever the filesystem hands back: alphabetically `dsocore`
    // precedes the `sgcore` it needs, and `arbfunc` precedes charinterface.
    // Rather than encode a dependency graph, just repeat the pass while any
    // module still succeeds; a module whose provider loaded in pass N opens
    // in pass N+1.  Failures are only reported once the passes stop making
    // progress, so a merely-out-of-order module never looks like an error.
    // (macOS does not need this — `-undefined dynamic_lookup` defers the
    // lookup — but an order-independent loader is right on every platform.)
    std::deque<XString> pending = modules;
    std::size_t prev_pending = 0;
    bool last_pass = false;
    while( !pending.empty()) {
    //! No module opened during the previous pass ⇒ nothing will change now;
    //! run one final pass that REPORTS the failures instead of deferring them.
    last_pass = (pending.size() == prev_pending);
    prev_pending = pending.size();
    std::deque<XString> retry;
    for(auto it = pending.begin(); it != pending.end(); it++) {
        app.processEvents(); //displays message.
        std::cerr <<  "Loading module \"" + *it + "\" " << std::endl;

        bool deprecated = false;
        XString lower_name = QString::fromStdString( *it).toLower().toStdString();
        for(const char *dep : deprecated_modules) {
            if(lower_name.find(dep) != std::string::npos) {
                deprecated = true;
                XMessageBox::post(
                    "Skipping deprecated module \"" + *it +
                    "\" — please remove the file from the modules directory.",
                    *g_pIconWarn);
                break;
            }
        }
        if(deprecated)
            continue;

#ifdef USE_LIBTOOL
        lt_dlhandle handle =
            lt_dlopenadvise(QString( *it).toLocal8Bit().data(), g_dl_advise);
#endif
#ifdef USE_LOADLIBRARY
        DWORD currerrmode = GetThreadErrorMode();
        SetThreadErrorMode(currerrmode | SEM_FAILCRITICALERRORS, NULL); //suppresses an error dialog on loading.
        HANDLE handle = LoadLibraryA(QString( *it).toLocal8Bit().data());
        DWORD lasterr = GetLastError();
        SetThreadErrorMode(currerrmode, NULL);
        SetLastError(lasterr);
#endif
        if(handle) {
            XMessageBox::post("Module \"" + *it + "\" loaded", *g_pIconKame);
            ++num_loaded_modules;
        }
        else {
            const char *why =
#ifdef USE_LIBTOOL
                lt_dlerror();
#else
                nullptr;
#endif
            if( !last_pass) {
                retry.push_back( *it);      //!< maybe a provider is not up yet
                continue;
            }
            // Also to stderr, and WITH the loader's reason.  The success path
            // above already logs to stderr, so a failure that only reached the
            // GUI message pane was the one event you could not see from a
            // terminal — and "module silently absent" surfaces much later as a
            // missing driver type or an unresolved Python name.
            std::cerr << "Failure during loading module \"" + *it + "\""
                      << (why ? XString(": ") + why : XString()) << std::endl;
            XMessageBox::post("Failure during loading module \"" + *it + "\"", *g_pIconError);
        }

    }
    if(last_pass) break;
    pending.swap(retry);
    }

    // All modules including Python modules are now loaded.
    // Signal the Python thread so it can proceed with xpythonsupport.py imports.
    form->signalAllModulesLoaded();

#if defined __MACOSX__ || defined __APPLE__
    //Disables App Nap
    suspendLazySleeps();
#endif
    XMessageBox::post(formatString_tr(I18N_NOOP("%d out of %d modules have been loaded."),
        num_loaded_modules, (int)modules.size()),
        (num_loaded_modules == (int)modules.size()) ? *g_pIconInfo : *g_pIconWarn);

    const char *greeting = "KAME ver:" VERSION ", built at " __DATE__ " " __TIME__;
    fprintf(stderr, "%s\n", greeting);
    gMessagePrint(greeting);

    //! RAII guard: pool stays active until guard goes out of scope at
    //! function return.  Replaces the historical bare `activateAllocator()`
    //! call by also handling `release_pools()` on shutdown.
    KamePooledAllocGuard pool_guard;

#ifndef USE_STD_ALLOCATOR
    //! (§35) KAME routinely cycles 100 MB-class image / waveform buffers
    //! through the large-recycle cache.  Raise its target total resident
    //! footprint to 3 GiB (from the ~2 GiB default) so a working set of a
    //! few dozen such buffers stays warm — each reuse then skips a fresh
    //! mmap + demand-zero fault on alloc and a munmap + TLB shootdown on
    //! free.  `total_bytes` is split internally ~half to the shared global
    //! L2 and ~half to the aggregate per-thread L1; set here at startup for
    //! an exact bound (per-thread L1 sizing is derived once when a thread
    //! first arms its cache, so a later change would not apply retroactively).
    //! Pool-only — `kame_pool_set_large_cache_cap` is declared in kame_pool.h,
    //! which is included solely on non-`USE_STD_ALLOCATOR` (i.e. pooled) builds.
    kame_pool_set_large_cache_cap((size_t)3 << 30);
#endif

    //! (§30) KAME is a measurement application — its hot path is tight
    //! instrument-control loops, not a server that needs aggressive RSS
    //! reclaim under bursty load.  Realtime-mode silences the allocator's
    //! three background maintenance paths (§28.1 lazy drain, §28.3 auto-
    //! tune startup probe, §21 thread-exit madvise) so they never inject
    //! a surprise munmap/madvise into a measurement loop.  The LRC_MMAP
    //! cap (`kame_pool_set_large_cache_cap`, set to 3 GiB just above) still
    //! applies — adjusting it is the supported way to bound RSS in
    //! realtime mode.
    kame_pool_set_realtime_mode(1);

#ifndef USE_STD_ALLOCATOR
    //! (§75) The per-thread half, process-wide at its CHEAP level.
    //! `KAME_RT_DEFER` stops every thread's `free()` from entering the kernel:
    //! chunk page-reclaim is skipped (the chunk stays immediately recyclable,
    //! its pages just stay warm) and a large-tier `munmap` is parked, bounded
    //! by `kame_pool_set_rt_pending_cap`.  All of that sits on cold release
    //! paths, so it costs nothing measurable, and it removes the spikes §30
    //! cannot reach — the ones a live thread's own free can still produce.
    //! Measured on the band the recycle cache cannot absorb (> 256 MiB):
    //! free median 128 ns vs 20,480 ns, max 792 ns vs 677,917 ns.
    //!
    //! Nothing drains the parked backlog here, deliberately: KAME has no
    //! natural "trough" like a control loop's inter-cycle gap, the cap bounds
    //! the VA, and any non-realtime large free settles one parked block on its
    //! way out.  A driver that wants a hard bound can call
    //! `kame_pool_rt_drain()` between acquisitions.
    //!
    //! `KAME_RT_STRICT` is deliberately NOT enabled — this is not an oversight
    //! to be "completed" later.  It additionally drops the cross-thread
    //! dealloc batch to per-free flushing, which measured **-47 %** of
    //! cross-thread small-free throughput (60.8 -> 32.6 M free/s, 8/8
    //! interleaved reps) because it gives up the batch's coalesced-CAS win —
    //! on exactly KAME's dominant pattern, an STM Payload cloned on one thread
    //! and released on another.  What it buys is the p99.9 mid-tail
    //! (96 ns vs 1,792 ns), which KAME cannot use: its deadlines are
    //! instrument I/O at millisecond scale.  A future driver with a genuine
    //! sub-millisecond software loop should call
    //! `kame_pool_set_realtime_thread(KAME_RT_STRICT)` on that thread alone.
    kame_pool_set_realtime_default(KAME_RT_DEFER);
#endif

#if defined __MACOSX__ || defined __APPLE__
    while(form->running()) {
        void *p = autoReleasePoolInit(); //may be needed to release OpenGL related objects.
        app.processEvents();
        autoReleasePoolRelease(p);
    }
    int ret = 0;
#else
    int ret = app.exec();
#endif

//#if defined __WIN32__ || defined WINDOWS || defined _WIN32
//    FreeConsole();
//#endif

	return ret;
}  
