rem @echo off
set QTROOT="C:\QT\"
:start
if not exist qtdir.txt (
 echo "Searching for QT in" %QTROOT% "...."
 dir /S/B %QTROOT% | findstr /r "mingw.*_32\\bin\\Qt5Core.dll" >qtdir.txt
)
set /p QTDIR= <qtdir.txt
if not exist %QTDIR% (
 echo "QT DLLs has lost...."
 del qtdir.txt
 goto start
)
set QTDIR=%QTDIR:\bin\Qt5Core.dll=%
set PATH=%QTDIR%\bin;%PATH%
kame.exe
