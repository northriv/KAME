@echo off
:start
set QTDIR=
if not exist qtdir.txt (
 if "%QTROOT%"=="C:\QT\" (set QTROOT=E:\QT\) else if "%QTROOT%"=="E:\QT\" (set QTROOT=C:\) else (set QTROOT=C:\QT\)
 echo Searching for QT DLLs in %QTROOT%....
 dir /S/B %QTROOT% | findstr /r "mingw.*_32\\bin\\Qt5Core.dll" >qtdir.txt
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
set QTDIR=%QTDIR:\bin\Qt5Core.dll=%
set PATH=%QTDIR%\bin;%PATH%
kame.exe --nooverpaint
