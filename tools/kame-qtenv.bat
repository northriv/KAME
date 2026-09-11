@echo off
rem Locate a Qt 6 (>= 6.5) llvm-mingw 64-bit kit and export QTDIR / PATH.
rem
rem Shared by kame.bat and kame-msyspython.bat, so the search exists once.
rem   call "%~dp0kame-qtenv.bat"        set QTDIR, QTTOOLDIR, QTMINGW64DIR, PATH
rem   call "%~dp0kame-qtenv.bat" mingw  the same, plus Qt's own mingw_64 kit on
rem                                     PATH. kame.bat needs libgcc_s_seh and
rem                                     api-ms-win-core-path from it; the MSYS2
rem                                     launcher takes those from msys64.
rem   kame-qtenv.bat print              report what it would use, change nothing
rem The "kame-" prefix is load-bearing: mkzip.bat packages a release with
rem `copy kame*.bat`, so a plainer name would be missing from the zip and every
rem launcher in it would die on the first line.
rem Sets errorlevel 1 when nothing is found; the callers stop there.
rem
rem The three things the older inline search got wrong, all user-visible:
rem  * `dir /S/B` from a drive root walked the whole disk on every uncached
rem    launch.  Qt installs as <root>\<version>\<kit>\bin, so two levels of
rem    globbing find it without a walk.
rem  * `set /p` then took the FIRST line, i.e. lexicographic order.  That
rem    preferred 6.10 over 6.9 only because "6.1" sorts before "6.9" -- and
rem    would just as happily prefer 6.10 over 6.20.  Versions are compared
rem    as numbers here.
rem  * On failure it cycled QTROOT and `goto start`, rescanning for ever; a
rem    machine with no Qt spun until killed.  This reports and returns.
setlocal enabledelayedexpansion

set "MODE=%~1"
set "QTCORE="
rem Cache beside this script, not in the current directory, so the search is
rem paid once per install rather than once per folder it is started from.
set "QTCACHE=%~dp0qtdir.txt"
set "BESTKEY=0"
set "BESTVER="

rem ---- 1. the cache, if it still points at something --------------------------
if exist "%QTCACHE%" (
    set /p QTCORE=<"%QTCACHE%"
    if not exist "!QTCORE!" (
        echo Cached Qt path no longer exists; searching again.
        del "%QTCACHE%" 2>nul
        set "QTCORE="
    )
)

rem ---- 2. the usual roots, two levels deep, no disk walk ----------------------
if not defined QTCORE (
    for %%R in ("%QTROOT%" "C:\Qt" "C:\QT" "D:\Qt" "E:\Qt" "%USERPROFILE%\Qt") do call :scan_root "%%~R"
    if defined QTCORE echo Found Qt !BESTVER!: "!QTCORE!"
)

rem ---- 3. last resort: ONE announced scan, whose hits feed the same picker ----
if not defined QTCORE (
    echo Qt was not in the usual places. Scanning C:\ once - this can take a
    echo few minutes. Set QTROOT to your Qt folder to skip it next time.
    for /f "delims=" %%L in ('dir /S/B "C:\Qt6Core.dll" 2^>nul ^| findstr /i /r /c:"llvm.*_64\\bin\\Qt6Core\.dll"') do (
        set "KIT=%%L"
        set "KIT=!KIT:\bin\Qt6Core.dll=!"
        for %%A in ("!KIT!\..\..") do call :scan_root "%%~fA"
    )
    if defined QTCORE echo Found Qt !BESTVER!: "!QTCORE!"
)

if not defined QTCORE (
    echo.
    echo Qt 6.5 or later ^(llvm-mingw, 64-bit^) was not found.
    echo Install it with the Qt online installer, or point QTROOT at the folder
    echo holding the version directories, e.g.   set QTROOT=D:\Qt
    endlocal
    exit /b 1
)

rem Remember it, so the next launch costs nothing.  A read-only install just
rem means the search happens again; it must not be fatal.
>"%QTCACHE%" echo !QTCORE! 2>nul

set "QTDIR=!QTCORE:\bin\Qt6Core.dll=!"
set "QTTOOLDIR="
set "QTMINGW64DIR="
for /d %%d in ("!QTDIR!\..\..\Tools\llvm-mingw*") do set "QTTOOLDIR=%%~fd"
rem mingw_64 is a sibling KIT holding a second, GCC-built Qt6Core.dll, so it
rem must stay behind QTDIR\bin and is only added when the caller asks -- that
rem is why it is opt-in rather than always on.
for %%d in ("!QTDIR!\..\mingw_64") do if exist "%%~fd\bin\" set "QTMINGW64DIR=%%~fd"

set "QTADD=!QTDIR!\bin"
if defined QTTOOLDIR set "QTADD=!QTADD!;!QTTOOLDIR!\bin"
if /i "%MODE%"=="mingw" if defined QTMINGW64DIR set "QTADD=!QTADD!;!QTMINGW64DIR!\bin"

if /i "%MODE%"=="print" (
    echo QTDIR=!QTDIR!
    echo QTTOOLDIR=!QTTOOLDIR!
    echo QTMINGW64DIR=!QTMINGW64DIR!
    echo PATH prefix=!QTADD!
    endlocal
    exit /b 0
)

rem endlocal discards everything set above, so hand the values across it.
endlocal & set "QTDIR=%QTDIR%" & set "QTTOOLDIR=%QTTOOLDIR%" & set "QTMINGW64DIR=%QTMINGW64DIR%" & set "PATH=%QTADD%;%PATH%"
exit /b 0

rem ---------------------------------------------------------------------------
rem :scan_root <folder>  - keep the highest-numbered >= 6.5 llvm-mingw kit under
rem it in QTCORE / BESTKEY / BESTVER.  Called for every candidate root, so the
rem comparison lives in exactly one place.
:scan_root
set "ROOT=%~1"
if "%ROOT%"=="" goto :eof
rem QTROOT is conventionally written with a trailing backslash; normalise so
rem the globs below never end up with "C:\Qt\\6.*".
for %%A in ("%ROOT%\.") do set "ROOT=%%~fA"
if not exist "%ROOT%\" goto :eof
for /d %%V in ("%ROOT%\6.*") do (
    rem Qt installs as 6.10.1 etc.; fold the three parts into one sortable
    rem number so "newer" is arithmetic, never lexicographic.
    for /f "tokens=1,2,3 delims=." %%a in ("%%~nxV") do (
        set "MAJ=%%a"
        set "MIN=%%b"
        set "PAT=%%c"
        if "!MIN!"=="" set "MIN=0"
        if "!PAT!"=="" set "PAT=0"
        set "KEY=0"
        set /a "KEY=!MAJ! * 1000000 + !MIN! * 1000 + !PAT!" 2>nul
        rem 6.5 is the floor: 6005000.
        if !KEY! GEQ 6005000 if !KEY! GTR !BESTKEY! (
            for /d %%K in ("%%~V\llvm*_64") do (
                if exist "%%~K\bin\Qt6Core.dll" (
                    set "BESTKEY=!KEY!"
                    set "BESTVER=%%~nxV"
                    set "QTCORE=%%~K\bin\Qt6Core.dll"
                )
            )
        )
    )
)
goto :eof
