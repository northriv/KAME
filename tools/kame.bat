rem @echo off
if not exist qtdir.txt (
 echo "Searching for QT...."
 dir /S/B C:\QT\ | findstr /r "mingw.*_32\\bin\\Qt5Core.dll" >qtdir.txt
)
set /p QTDIR= <qtdir.txt
set QTDIR=%QTDIR:\bin\Qt5Core.dll=%
set PATH=%QTDIR%\bin;%PATH%
kame.exe
