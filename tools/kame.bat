@echo off
rem KAME launcher - Qt 6 llvm-mingw + the bundled CPython under .\resources.

call "%~dp0kame-qtenv.bat" mingw
if errorlevel 1 (
    pause
    exit /b 1
)
echo Qt: %QTDIR%

rem `unset` is a Unix shell builtin, not a cmd command: the line that used to
rem be here printed "not recognized" and left any inherited PYTHONHOME in
rem place, pointing the bundled interpreter -- and the MCP server started from
rem it -- at a foreign standard library.
set "PYTHONHOME="
set "PYTHONPATH=.\resources\python3.12;.\resources\python3.12\site-packages;.\resources\python3.12\lib-dynload"

rem To inspect the DLL closure:  C:\msys64\usr\bin\ldd.exe kame.exe
kame.exe %*
