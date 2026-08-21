@echo off

:start

set QTDIR=

if not exist qtdir.txt (

 if "%QTROOT%"=="C:\QT\" (set QTROOT=E:\QT\) else if "%QTROOT%"=="E:\QT\" (set QTROOT=C:\) else (set QTROOT=C:\QT\)

 echo Searching for QT 6.5 or later DLLs in %QTROOT%....

rem Two patterns, because findstr has no alternation and `*` repeats only the
rem PRECEDING element: "6.[5-9]" alone matches 6.5-6.9 but NOT a two-digit
rem minor, so Qt 6.10/6.11 went undetected here while kame.bat (plain "6.*")
rem found them.  The second /c: covers 6.10 and later; together they keep the
rem original ">= 6.5" intent.
 dir /S/B %QTROOT% | findstr /r /c:"6.[5-9].*\\llvm.*64\\bin\\Qt6Core.dll" /c:"6.[1-9][0-9].*\\llvm.*64\\bin\\Qt6Core.dll" >qtdir.txt

)

set /p QTDIR= <qtdir.txt

if "%QTDIR%"=="" (

 echo "QT DLLs not found...."

 del qtdir.txt

 goto start

)

if not exist %QTDIR% (

 echo "QT DLLs has lost...."

 del qtdir.txt

 goto start

)



set QTDIR=%QTDIR:\bin\Qt6Core.dll=%
for /d %%d in (%QTDIR%\..\..\Tools\llvm-mingw*) do @set QTTOOLDIR=%%d
echo %QTTOOLDIR%

set PATH=%QTDIR%\bin;%QTTOOLDIR%\bin;%PATH%
set PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;C:\msys64\mingw64\lib;%PATH%
set PYTHONHOME=C:\msys64\mingw64

set PYTHONPATH=C:\msys64\mingw64\lib\python3.12;C:\msys64\mingw64\lib\python3.12\site-packages;C:\msys64\mingw64\lib\python3.12\lib-dynload

kame.exe

