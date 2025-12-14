@echo off

:start

set QTDIR=

if not exist qtdir.txt (

 if "%QTROOT%"=="C:\QT\" (set QTROOT=E:\QT\) else if "%QTROOT%"=="E:\QT\" (set QTROOT=C:\) else (set QTROOT=C:\QT\)

 echo Searching for QT 6.5 or later DLLs in %QTROOT%....

 dir /S/B %QTROOT% | findstr /r "6.*\\llvm.*64\\bin\\Qt6Core.dll" >qtdir.txt

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
for /d %%d in (%QTDIR%\..\mingw_64) do @set QTMINGW64DIR=%%d
echo %QTMINGW64DIR% #for libgcc_s_seh and api-ms-win-core-path

set PATH=%QTDIR%\bin;%QTTOOLDIR%\bin;%QTMINGW64DIR%\bin;%PATH%

unset PYTHONHOME

set PYTHONPATH=.\resources\python3.12;.\resources\python3.12\site-packages;.\resources\python3.12\lib-dynload

#C:\msys64\usr\bin\ldd.exe kame.exe

kame.exe

