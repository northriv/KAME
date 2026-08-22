TARGET = kame
TEMPLATE = app

PRI_DIR = ../
include(../kame.pri)

macx: SCRIPT_DIR = Resources
win32: SCRIPT_DIR = resources
# Linux/BSD: the deployed scripts sit next to the executable in a build tree
# (and in $$PREFIX/share/kame once installed, which QStandardPaths finds on
# its own).  Without this, LINESHELL_DIR expanded to a bare "/" and
# FrmKameMain::scriptLineShellAction_activated() could never locate
# rubylineshell.rb / pythonlineshell.py.
unix:!macx: SCRIPT_DIR = .
DEFINES += LINESHELL_DIR=\"quotedefined($${SCRIPT_DIR}/)\"
DEFINES += USE_STD_RANDOM

CONFIG += CONSOLE

#win32: QMAKE_CXXFLAGS += -pie

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}\
    $${_PRO_FILE_PWD_}/../kamepoolalloc\
    $${_PRO_FILE_PWD_}/../kamestm\
    $${_PRO_FILE_PWD_}/math\
    $${_PRO_FILE_PWD_}/forms\
    $${_PRO_FILE_PWD_}/thermometer\
    $${_PRO_FILE_PWD_}/analyzer\
    $${_PRO_FILE_PWD_}/driver\
    $${_PRO_FILE_PWD_}/graph\
    $${_PRO_FILE_PWD_}/script\
    $${_PRO_FILE_PWD_}/icons

# The Ruby INTERPRETER, dropped by `qmake CONFIG+=no_ruby` (see kame.pri).
# xrubywriter.* is NOT here: it only writes text, needs no libruby, and the
# .kam format depends on it.
RUBY_HEADERS =
RUBY_SOURCES =
!no_ruby {
    RUBY_HEADERS = script/xrubysupport.h script/rubywrapper.h
    RUBY_SOURCES = script/xrubysupport.cpp script/rubywrapper.cpp
}

HEADERS += \
    ../kamepoolalloc/allocator.h \
    ../kamepoolalloc/allocator_prv.h \
    ../kamepoolalloc/atomic.h \
    ../kamepoolalloc/atomic_mfence.h \
    ../kamepoolalloc/atomic_smart_ptr.h \
    ../kamepoolalloc/kame_pool.h \
    graph/graphmathfittool.h \
    graph/graphmathtool.h \
    graph/graphmathtoolconnector.h \
    graph/graphntoolbox.h \
    graph/onscreenobject.h \
    graph/x2dimage.h \
    kame.h \
    script/xscriptingthread.h \
    script/xscriptingthreadconnector.h \
    ../kamestm/threadlocal.h \
    ../kamestm/transaction_impl.h \
    ../kamestm/transaction_signal.h \
    ../kamestm/transaction.h \
    ../kamestm/transaction_detail.h \
    ../kamestm/transaction_negotiation.h \
    ../kamestm/transaction_neg_impl.h \
    ../kamestm/transaction_definitions.h \
    ../kamestm/xthread.h \
    ../kamestm/xwaitcell.h \
    ../kamestm/xtime.h \
    ../kamestm/atomic_queue.h \
    ../kamestm/fast_vector.h \
    driver/driver.h \
    driver/dummydriver.h \
    driver/interface.h \
    driver/primarydriver.h \
    driver/primarydriverwiththread.h \
    driver/secondarydriver.h \
    driver/secondarydriverinterface.h \
    driver/softtrigger.h \
    graph/graph.h \
    graph/graphdialogconnector.h \
    graph/graphpainter.h \
    graph/graphwidget.h \
    graph/xwavengraph.h \
    analyzer/analyzer.h \
    analyzer/recorder.h \
    analyzer/recordreader.h \
    script/xdotwriter.h \
    $$RUBY_HEADERS \
    script/xrubywriter.h \
    xitemnode.h \
    xlistnode.h \
    xnode.h \
    xnodeconnector.h \
    xscheduler.h \
    icons/icon.h \
    measure.h \
    support.h \
    thermometer/caltable.h \
    thermometer/thermometer.h \
    math/ar.h \
    math/cspline.h \
    math/fft.h \
    math/fir.h \
    math/freqestleastsquare.h \
    math/rand.h \
    math/spectrumsolver.h \
    forms/driverlistconnector.h \
    forms/entrylistconnector.h \
    forms/graphlistconnector.h \
    forms/calibentryconnector.h \
    forms/interfacelistconnector.h \
    forms/nodebrowser.h \
    forms/recordreaderconnector.h \
    messagebox.h \
    math/nllsfit.h \
    math/tikhonovreg.h

SOURCES += icons/icon.cpp \
    graph/graphmathtool.cpp \
    graph/graphmathtoolconnector.cpp \
    graph/graphntoolbox.cpp \
    graph/onscreenobject.cpp \
    graph/x2dimage.cpp \
    script/xscriptingthread.cpp \
    script/xscriptingthreadconnector.cpp \
    ../kamestm/xthread.cpp \
    ../kamestm/xtime.cpp \
    support.cpp \
    graph/graphdialogconnector.cpp \
    graph/graphpainter.cpp \
    graph/graphpaintergl.cpp \
    graph/graphwidget.cpp \
    graph/xwavengraph.cpp \
    graph/graph.cpp \
    thermometer/caltable.cpp \
    thermometer/thermometer.cpp \
    xitemnode.cpp \
    xlistnode.cpp \
    xscheduler.cpp \
    math/ar.cpp \
    math/cspline.cpp \
    math/fft.cpp \
    math/fir.cpp \
    math/freqestleastsquare.cpp \
    math/rand.cpp \
    math/spectrumsolver.cpp \
    script/xdotwriter.cpp \
    $$RUBY_SOURCES \
    script/xrubywriter.cpp \
    measure.cpp \
    ../kamestm/threadlocal.cpp \
    xnode.cpp \
    xnodeconnector.cpp \
    driver/driver.cpp \
    driver/interface.cpp \
    driver/primarydriver.cpp \
    driver/secondarydriver.cpp \
    driver/softtrigger.cpp \
    forms/driverlistconnector.cpp \
    forms/entrylistconnector.cpp \
    forms/graphlistconnector.cpp \
    forms/calibentryconnector.cpp \
    forms/interfacelistconnector.cpp \
    forms/nodebrowser.cpp \
    forms/recordreaderconnector.cpp \
    analyzer/analyzer.cpp \
    analyzer/recorder.cpp \
    analyzer/recordreader.cpp\
    kame.cpp \
    main.cpp \
    messagebox.cpp \
    math/tikhonovreg.cpp

# (Production kame.exe inline-compiles kamepoolalloc's allocator.cpp on
# every platform — same model as the kamestm sources xthread.cpp /
# xtime.cpp / threadlocal.cpp listed in the top-level SOURCES above.
# No DLL boundary at runtime: kamepoolalloc.pro's libkamepoolalloc.dll
# exists only for the test scaffold.  The `__DATA,__interpose` section
# emitted by allocator.cpp is dead data inside an MH_EXECUTE image —
# dyld only honours interpose from MH_DYLIB — so the inline path is
# functionally identical to the previous in-kame `kame/allocator.cpp`
# layout.)
#
# CAUTION, LINUX: the "interpose is inert in the executable" reasoning
# above is Mach-O-specific and does NOT carry over to ELF.  allocator.cpp's
# `#elif defined(__linux__)` block emits malloc / free / calloc /
# posix_memalign / aligned_alloc / memalign as ordinary strong symbols,
# and the executable is first in ELF's global symbol scope — so with
# `-rdynamic` (added below for the ltdl modules) the kame binary becomes
# the process-wide allocator for Qt, Mesa, libpython, libruby, libusb,
# libgsl and everything else, not just for KAME's own new/delete.  That is
# the same configuration kamepoolalloc is soaked in under LD_PRELOAD
# (mimalloc-bench), so it is believed sound, but it IS a different runtime
# shape from macOS and Windows.  Build with
# `DEFINES += KAMEPOOLALLOC_NO_LIBC_INTERPOSE` to get macOS-like parity
# (operator new/delete pooled, libc malloc untouched).
SOURCES += ../kamepoolalloc/allocator.cpp

unix {
    HEADERS += \
        math/matrix.h \
        math/freqest.h
    # Boost-ublas matrix helpers, Unix-only (boost isn't part of the
    # standard MacPorts set documented in CLAUDE.md but is provided by
    # distro packages on Linux; not pulled in on Windows).
    SOURCES += \
        math/freqest.cpp \
        math/matrix.cpp
}

FORMS += \
    forms/caltableform.ui \
    forms/drivercreate.ui \
    forms/drivertool.ui \
    forms/graphtool.ui \
    forms/interfacetool.ui \
    forms/nodebrowserform.ui \
    forms/recordreaderform.ui \
    forms/scalarentrytool.ui \
    forms/messageform.ui \
    forms/scriptingthreadtool.ui

RESOURCES += \
    kame.qrc

DESTDIR=$$OUT_PWD/../
# On Linux/BSD the target is a bare executable called `kame`, and the build
# tree already contains a DIRECTORY called `kame` (this subproject) in exactly
# that place — so `ld` fails with "cannot open output file ...: Is a
# directory".  macOS escapes it because the target is the `kame.app` bundle
# and Windows because it is `kame.exe`; only the Unix name collides.  Put the
# executable one level down instead.
unix:!macx: DESTDIR = $$OUT_PWD/../bin

scriptfile.files = script/rubylineshell.rb \
    script/pythonlineshell.py \
    script/kame_mcp_server.py \
    script/kame_pydantic_ai.py \
    script/kame_python_api.md \
    ../doc/manual/kame-8-en.md \
    script/notebook/jupyter_notebook_config.py \
    script/notebook/notebook_kame_kernel_manager.py

macx {
    scriptfile.path = Contents/Resources
    QMAKE_BUNDLE_DATA += scriptfile

    # The Claude Code plugin (kame skill + MCP server launcher), copied as a
    # whole directory to Contents/Resources/plugin.  The kame:claude-cli
    # quick-launch link passes it to `claude --plugin-dir`, and the plugin's
    # own launcher finds kame_mcp_server.py right above it at ../ .
    pluginfiles.files = script/plugin
    pluginfiles.path = Contents/Resources
    QMAKE_BUNDLE_DATA += pluginfiles

    LIBS += -L$$OUT_PWD/ -llibkame
}
else {
    #in macx, these are in libkame
    FORMS += \
        graph/graphdialog.ui \
        graph/graphform.ui \
        graph/graphnurlform.ui
    SOURCES +=\
        icons/kame-24x24-png.c

    unix {
        # `scriptfile.path` was only ever set inside the macx branch above, so
        # this INSTALLS entry produced nothing but qmake's "scriptfile.path is
        # not defined: install target not created" warning.  Install to
        # $$PREFIX/share/kame, which is where QStandardPaths::AppDataLocation
        # looks for applicationName "kame".
        isEmpty(PREFIX): PREFIX = /usr/local
        scriptfile.files += ../kame_ja.qm     # main.cpp looks next to the binary
        # Thamway EZ-USB firmware / GPIF images.  These were deployed only via
        # the macx QMAKE_BUNDLE_DATA block below, yet libthamway.so does build
        # on Linux — so opening the interface failed with "USB GPIF/firmware
        # file fx2fw.bix not found" and there was nowhere the build had put
        # it.  XCyFXUSBInterface looks in QStandardPaths::AppDataLocation
        # (= $$PREFIX/share/kame) and then applicationDirPath(); the staging
        # loop below covers the second.
        exists(../modules/nmr/thamway/fx2fw.bix) {
            scriptfile.files += ../modules/nmr/thamway/fx2fw.bix \
                ../modules/nmr/thamway/slow_dat.bin \
                ../modules/nmr/thamway/fullspec_dat.bin
        }
        scriptfile.path = $${PREFIX}/share/kame
        INSTALLS += scriptfile
        # The Claude Code plugin directory, whole (see the macx block).
        pluginfiles.files = script/plugin
        pluginfiles.path = $${PREFIX}/share/kame
        INSTALLS += pluginfiles
        # Also stage them beside the binary so an uninstalled build tree is
        # directly runnable — the equivalent of QMAKE_BUNDLE_DATA on macOS.
        for(f, scriptfile.files): \
            QMAKE_POST_LINK += $$quote(cp -f $${_PRO_FILE_PWD_}/$${f} $${DESTDIR}/ &&) \

        QMAKE_POST_LINK += $$quote(cp -Rf $${_PRO_FILE_PWD_}/script/plugin $${DESTDIR}/ &&)
        QMAKE_POST_LINK += true

        # The executable itself was never in INSTALLS, so `make install`
        # deployed data files and no program.  (macOS installs the .app
        # bundle, Windows copies by hand.)
        target.path = $${PREFIX}/bin
        INSTALLS += target

        # Desktop integration (Linux-only files that nothing ever installed).
        desktopfile.files = kame.desktop
        desktopfile.path = $${PREFIX}/share/applications
        INSTALLS += desktopfile

        # udev rules for the libusb instrument drivers.  Not installed into
        # /etc by default (a --prefix build must not write outside its prefix);
        # ship them where a packager or the user can pick them up.
        udevrules.files = 70-kame.rules
        udevrules.path = $${PREFIX}/lib/udev/rules.d
        INSTALLS += udevrules
        # The PNGs are named hi{16,32}-app-kame.png (the old KDE icon naming);
        # the hicolor theme requires the basename to equal the `Icon=` key, so
        # install them renamed via .extra rather than .files.  (Written out
        # twice rather than looped: qmake's for() cannot assign to a computed
        # variable name without eval(), and silently does nothing.)
        icon16.path = $${PREFIX}/share/icons/hicolor/16x16/apps
        icon16.extra = \
            mkdir -p $(INSTALL_ROOT)$${PREFIX}/share/icons/hicolor/16x16/apps && \
            $(INSTALL_FILE) $${_PRO_FILE_PWD_}/hi16-app-kame.png \
                $(INSTALL_ROOT)$${PREFIX}/share/icons/hicolor/16x16/apps/kame.png
        icon16.CONFIG += no_check_exist
        icon32.path = $${PREFIX}/share/icons/hicolor/32x32/apps
        icon32.extra = \
            mkdir -p $(INSTALL_ROOT)$${PREFIX}/share/icons/hicolor/32x32/apps && \
            $(INSTALL_FILE) $${_PRO_FILE_PWD_}/hi32-app-kame.png \
                $(INSTALL_ROOT)$${PREFIX}/share/icons/hicolor/32x32/apps/kame.png
        icon32.CONFIG += no_check_exist
        exists($${_PRO_FILE_PWD_}/hi16-app-kame.png): INSTALLS += icon16
        exists($${_PRO_FILE_PWD_}/hi32-app-kame.png): INSTALLS += icon32
    }
    else {
        # Keep in step with scriptfile.files above and with
        # tools/deploy_scripts.bat, which is what actually copies on Windows.
        # This list only makes the files visible in the IDE, so an omission
        # here is invisible -- and a tempting template for the next addition.
        DISTFILES += script/rubylineshell.rb  \
            script/pythonlineshell.py \
            script/kame_mcp_server.py \
            script/kame_pydantic_ai.py \
            script/kame_python_api.md \
            ../doc/manual/kame-8-en.md \
            script/notebook/jupyter_notebook_config.py \
            script/notebook/notebook_kame_kernel_manager.py

        # DISTFILES only lists files for the IDE -- it copies nothing, so the
        # Windows build used to leave $$DESTDIR/resources without any of them
        # and kame.exe started with no kame_mcp_server.py beside it (the MCP
        # link then died with "can't open file ...\Resources\
        # kame_mcp_server.py").  Deploy them at link time, the way the macOS
        # bundle and the Linux QMAKE_POST_LINK above already do.  The work
        # lives in a batch file rather than inline qmake so the quoting stays
        # legible and it can be run by hand (tools/mkzip.bat uses it too).
        # system_path(), not shell_path(): with MSYS on PATH qmake decides the
        # make shell is sh and shell_path() emits /C/Users/... , which the
        # recipe -- run under `mingw32-make SHELL=cmd.exe`, as this project is
        # built -- cannot execute.  system_path() gives native C:\Users\... .
        QMAKE_POST_LINK += $$quote(cmd /c $$system_path($${_PRO_FILE_PWD_}/../tools/deploy_scripts.bat) $$system_path($${DESTDIR}/$${SCRIPT_DIR}))
    }
}

#win32: QMAKE_POST_LINK += $$quote(cmd /c copy /y $${_PRO_FILE_PWD_}$${scriptfile.files} $${DESTDIR}$${SCRIPT_DIR}$$escape_expand(\\n\\t))

macx: ICON = kame.icns

#Ruby, pybind11
macx {
  !no_ruby {
    exists("/opt/local/include/ruby-*") {
        #for macports ruby3
        RUBYH = $$files("/opt/local/include/ruby-*")
        INCLUDEPATH += $${RUBYH}
        INCLUDEPATH += $${RUBYH}/arm64-darwin23
        INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/Current/Headers
        INCLUDEPATH += /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/
        LIBS += $$files(/opt/local/lib/libruby.*.dylib)
        message("using ruby from macports.")
    }
    else {
        INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/Current/Headers
        INCLUDEPATH += /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Ruby.framework/Versions/Current/Headers/
        LIBS += -framework Ruby
    #for ruby.h incompatible with C++11
        QMAKE_CXXFLAGS += -Wno-error=reserved-user-defined-literal
        message("using framework ruby.")
    }
  }

    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="python3" $$files("/opt/local/bin/python3*") $$files("/usr/local/bin/python3*")
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
    #            QMAKE_CXXFLAGS += $$system("$${PYTHON}-config --cflags")
                QMAKE_LFLAGS += $$system("$${PYTHON}-config --embed --ldflags")
                DEFINES += USE_PYBIND11
                DEFINES += PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF #For mainthread call.
                SOURCES += script/xpythonmodule.cpp \
                    script/xpythonsupport.cpp
                HEADERS += script/xpythonmodule.h \
                    script/xpythonsupport.h \
                    driver/pythondriver.h
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
else:unix {
    # Linux/BSD.  The previous hard-coded `/usr/lib/ruby/1.8/i386-linux/`
    # was a Ruby-1.8, 32-bit-x86 path and had not existed on any current
    # distribution for many years; ask the interpreter instead, exactly as
    # the macOS branch above globs MacPorts.  `rubyhdrdir` holds ruby.h and
    # `rubyarchhdrdir` the per-arch ruby/config.h — BOTH are required.
  !no_ruby {
    RUBY_BIN = $$system(which ruby)
    !isEmpty(RUBY_BIN) {
        RUBY_HDRDIR = $$system($${RUBY_BIN} -rrbconfig -e \'print RbConfig::CONFIG[\"rubyhdrdir\"]\')
        RUBY_ARCHHDRDIR = $$system($${RUBY_BIN} -rrbconfig -e \'print RbConfig::CONFIG[\"rubyarchhdrdir\"]\')
        RUBY_LIBDIR = $$system($${RUBY_BIN} -rrbconfig -e \'print RbConfig::CONFIG[\"libdir\"]\')
        RUBY_SONAME = $$system($${RUBY_BIN} -rrbconfig -e \'print RbConfig::CONFIG[\"RUBY_SO_NAME\"]\')
    }
    exists($${RUBY_HDRDIR}/ruby.h) {
        INCLUDEPATH += $${RUBY_HDRDIR} $${RUBY_ARCHHDRDIR}
        LIBS += -L$${RUBY_LIBDIR} -l$${RUBY_SONAME}
        # `-L` is link-time only for GNU ld — nothing is recorded in the ELF.
        # Without a RUNPATH the binary dies at exec with "libruby.so.N: cannot
        # open shared object file" for every rbenv/rvm/MacPorts-style Ruby,
        # i.e. exactly the non-system installs this RbConfig probe exists to
        # support.  macOS is immune (dylibs carry an install_name).  Skip the
        # standard system dirs so distro packages stay RUNPATH-free.
        !contains(RUBY_LIBDIR, "^/usr/lib.*"): !equals(RUBY_LIBDIR, /lib): \
            QMAKE_RPATHDIR += $${RUBY_LIBDIR}
        message("using ruby headers from $${RUBY_HDRDIR}.")
    }
    else {
        error("No Ruby development headers found (install ruby-dev / ruby-devel), \
or build without the Ruby interpreter: qmake CONFIG+=no_ruby")
    }
  }

    # Python / pybind11.  The macOS and win32-g++ branches each grow their
    # own copy of this block; Linux never had one, so USE_PYBIND11 was never
    # defined here — which silently disabled the Python scripting engine, the
    # Jupyter/IPython console, the MCP server AND the preferred .kam loader
    # (xrubysupport is then the only reader left).  Same probe as the others.
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons=$$system(which python3) $$files("/usr/bin/python3.[0-9]") $$files("/usr/bin/python3.[0-9][0-9]")
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes > /dev/null 2>&1") {
                # Take the LINKER flags from the versioned `pythonX.Y-config`
                # belonging to this very interpreter, never from a bare
                # `python3-config`.  On Linux those are separate alternatives
                # and routinely disagree — on this host `python3` is 3.11
                # while `python3-config` reports 3.12 — which yields headers
                # from one version linked against the library of another.
                PYVER = $$system("$${PYTHON} -c \'import sys; print(\"%d.%d\" % sys.version_info[:2])\'")
                PYCFG = $$dirname(PYTHON)/python$${PYVER}-config
                !exists($${PYCFG}): PYCFG = python$${PYVER}-config
                system("$${PYCFG} --embed --ldflags > /dev/null 2>&1") {
                    QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
                    # LIBS, not QMAKE_LFLAGS: qmake emits QMAKE_LFLAGS BEFORE
                    # the object files, and GNU ld resolves left-to-right with
                    # --as-needed on by default, so -lpython3.x placed there is
                    # discarded and every Py* symbol comes out undefined.  The
                    # macOS branch gets away with QMAKE_LFLAGS; GNU ld does not.
                    LIBS += $$system("$${PYCFG} --embed --ldflags")
                    # Same RUNPATH problem as Ruby above: a pyenv/conda
                    # libpython3.x.so is only found at run time if its
                    # directory is recorded in the ELF.
                    PYLIBDIR = $$system("$${PYTHON} -c \'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\") or \"\")\'")
                    !isEmpty(PYLIBDIR): !contains(PYLIBDIR, "^/usr/lib.*"): !equals(PYLIBDIR, /lib): \
                        QMAKE_RPATHDIR += $${PYLIBDIR}
                    DEFINES += USE_PYBIND11
                    DEFINES += PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF #For mainthread call.
                    SOURCES += script/xpythonmodule.cpp \
                        script/xpythonsupport.cpp
                    HEADERS += script/xpythonmodule.h \
                        script/xpythonsupport.h \
                        driver/pythondriver.h
                    message("Python scripting support enabled ($${PYTHON}, $${PYCFG}).")
                    break()
                }
            }
        }
        !contains(DEFINES, USE_PYBIND11): \
            message("pybind11 not found for any python3 — Python scripting, \
Jupyter and the MCP server are DISABLED, and .kam files fall back to the Ruby loader.")
    }
}
win32-*g++ {
  !no_ruby {
    exists($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include/ruby.h) {
    #for user-build ruby
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/i386-mingw32
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/x64-mingw64
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/x64-mingw32
        LIBS += $$files($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/lib*msvcrt-ruby*[0-9].dll.a)
        message("using ruby from ../ruby.")
    }
    else {
    #for msys64 ruby
        RUBYH = $$files("c:/msys64/mingw64/include/ruby-*")
        INCLUDEPATH += $${RUBYH}
        INCLUDEPATH += $${RUBYH}/x64-mingw32
        LIBS += $$files(c:/msys64/mingw64/lib/libx64-msvcrt-ruby*[0-9].dll.a)
        message("using ruby from msys2.")
    }
  }
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="c:/msys64/mingw64/bin/python.exe"
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
    #            QMAKE_CXXFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --cflags\"")
        #        QMAKE_LFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --embed --ldflags\"")
                LIBS += $$files(c:/msys64/mingw64/lib/libpython3*)
                DEFINES += USE_PYBIND11
                DEFINES += PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF #For mainthread call.
                SOURCES += script/xpythonmodule.cpp \
                    script/xpythonsupport.cpp
                HEADERS += script/xpythonmodule.h \
                    script/xpythonsupport.h \
                    driver/pythondriver.h
                message("Python scripting support enabled.")
                break()
            }
        }
    }
    LIBS += -lopengl32 -lglu32
}
win32-msvc* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/i386-mswin32_120
    !exists($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/libmsvcr*-ruby2*[0-9].lib) {
        error("No Ruby2 library!")
    }
    LIBS += $$files($${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/libmsvcr*-ruby2*[0-9].lib)
#    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby -lmsvcr120-ruby212 #-static -lWS2_32 -lAdvapi32 -lShell32 -limagehlp -lShlwapi -lIphlpapi
}

win32 {
    contains(QMAKE_HOST.arch, x86_64) {
        LIBS += -lz
    }
    else {
        INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../zlib/include
        LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../zlib/lib
        LIBS += -lzdll
    }
}
win32-msvc* {
    QMAKE_PRE_LINK += lib /machine:x86 /def:$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3/libfftw3-3.def /out:$${_PRO_FILE_PWD_}/$${PRI_DIR}../fftw3/libfftw3-3.lib
    QMAKE_PRE_LINK += & lib /machine:x86 /def:$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/libgsl.def /out:$${_PRO_FILE_PWD_}/$${PRI_DIR}../gsl/libgsl.lib
}

unix {
#    LIBS += -lclapack -lcblas -latlas
    macx {
        LIBS += -lfftw3
        LIBS += -lz
    }
    else {
        PKGCONFIG += fftw3
        PKGCONFIG += zlib
        # GLU (gluProject / gluUnProject in graphpaintergl.cpp).  macOS gets
        # it from the OpenGL framework and win32-g++ links -lglu32 below;
        # Linux/BSD needs it named explicitly.
        PKGCONFIG += glu
    }
    LIBS += -lltdl
}

#exports symbols from the executable for plugins.
macx {
  QMAKE_LFLAGS += -all_load -dynamic
}
unix:!macx {
  # The Linux counterpart of the two branches below, and it was missing.
  # By default GNU ld puts only what the executable itself needs into
  # .dynsym (40 defined symbols, none of them KAME's), so every module
  # carries unresolved references to the app — including DATA symbols such
  # as `XDriverList::s_types` and Transactional::Node<XNode>'s statics.
  # Modules still `dlopen` under RTLD_LAZY, which is what makes this so
  # easy to miss: they load, and then either fail at first call or, worse,
  # bind to a private per-.so copy of a singleton the whole design assumes
  # is process-wide (the type registries, the STM node statics, the pool
  # allocator's region list).  --export-dynamic is what makes the
  # executable a real symbol provider for its plugins.
  QMAKE_LFLAGS += -rdynamic
}
win32-g++ {
  QMAKE_LFLAGS += -Wl,--export-all-symbols -Wl,--out-implib,$${TARGET}.a #failed in debug config. cannot hold all of debug symbols.
}
win32-clang-g++ {
  QMAKE_LFLAGS += -Wl,--out-implib,$${TARGET}.a
  DEFINES += DECLSPEC_KAME=__declspec(dllexport)
  DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
  DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
}
win32-msvc* {
    DEFINES += DECLSPEC_KAME=__declspec(dllexport)
    DEFINES += DECLSPEC_MODULE=__declspec(dllexport)
    DEFINES += DECLSPEC_SHARED=__declspec(dllexport)
    # /utf-8: kame.pro inline-compiles kamepoolalloc/allocator.cpp, which
    # contains UTF-8 box-drawing characters in comments.  Without /utf-8,
    # MSVC on Japanese Windows (CP932 system code page) may swallow newlines
    # inside multi-byte character sequences, hiding #define directives and
    # causing C2065 "undeclared identifier" errors on macros in those files.
    QMAKE_CXXFLAGS += /utf-8
}

macx {
    HEADERS += \
        support_osx.h

    OBJECTIVE_SOURCES += \
        support_osx.mm

    LIBS += -framework Foundation

    coremodulefiles.files += ../modules/charinterface/libcharinterface.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/dcsource/core/libdcsourcecore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/dmm/core/libdmmcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/flowcontroller/core/libflowcontrollercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/levelmeter/core/liblevelmetercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/magnetps/core/libmagnetpscore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/motor/core/libmotorcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/relay/core/librelaycore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/networkanalyzer/core/libnetworkanalyzercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/nmr/pulsercore/libnmrpulsercore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/sg/core/libsgcore.$${QMAKE_EXTENSION_SHLIB}
    coremodulefiles.files += ../modules/lia/core/libliacore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/dso/core/libdsocore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/qd/core/libqdcore.$${QMAKE_EXTENSION_SHLIB}
    coremodule2files.files += ../modules/optics/core/libopticscore.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/testdriver/libtestdriver.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/counter/libcounter.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/dcsource/libdcsource.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/dmm/libdmm.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/dso/libdso.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/flowcontroller/libflowcontroller.$${QMAKE_EXTENSION_SHLIB}
#    modulefiles.files += ../modules/fourres/libfourres.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/funcsynth/libfuncsynth.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/levelmeter/liblevelmeter.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/lia/liblia.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/magnetps/libmagnetps.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/montecarlo/libmontecarlo.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/motor/libmotor.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/networkanalyzer/libnetworkanalyzer.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/nidaq/libnidaq.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/nmr/libnmr.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/nmr/libnmrpulser.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/sg/libsg.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/tempcontrol/libtempcontrol.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/nmr/thamway/libthamway.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/qd/libqd.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/digilentwf/libdigilentwf.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/gauge/libgauge.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/pumpcontroller/libpumpcontroller.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/arbfunc/libarbfunc.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/optics/liboptics.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/twoaxis/libtwoaxis.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/relay/librelay.$${QMAKE_EXTENSION_SHLIB}
    modulefiles.files += ../modules/python/libpython.$${QMAKE_EXTENSION_SHLIB}

    coremodulefiles.path = Contents/MacOS/$${KAME_COREMODULES}
    QMAKE_BUNDLE_DATA += coremodulefiles
    coremodule2files.path = Contents/MacOS/$${KAME_COREMODULES2}
    QMAKE_BUNDLE_DATA += coremodule2files
    modulefiles.path = Contents/MacOS/$${KAME_MODULES}
    QMAKE_BUNDLE_DATA += modulefiles

    tsfiles.files += ../kame_ja.qm
    tsfiles.path = Contents/MacOS/
    QMAKE_BUNDLE_DATA += tsfiles

    QMAKE_INFO_PLIST = ../Info.plist

#    exists("/opt/local/include/libusb-1.0/libusb.h") {
    exists("../modules/nmr/thamway/fx2fw.bix") {
        ezusbfiles.path = Contents/Resources
        ezusbfiles.files += ../modules/nmr/thamway/fx2fw.bix
        ezusbfiles.files += ../modules/nmr/thamway/slow_dat.bin
        ezusbfiles.files += ../modules/nmr/thamway/fullspec_dat.bin
        QMAKE_BUNDLE_DATA += ezusbfiles
    }
}



