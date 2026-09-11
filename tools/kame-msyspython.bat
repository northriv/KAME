@echo off
rem KAME launcher - Qt 6 llvm-mingw + MSYS2's mingw64 CPython, which is the
rem one that can carry ipykernel/numpy and therefore the in-process Jupyter
rem kernel the MCP server attaches to.

call "%~dp0kame-qtenv.bat"
if errorlevel 1 (
    pause
    exit /b 1
)
echo Qt: %QTDIR%

set "PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;C:\msys64\mingw64\lib;%PATH%"
rem Deliberate, and unlike kame.bat: this build IS the MSYS2 interpreter.
rem Anything spawned from here that is a real CPython (the MCP server) must
rem strip PYTHONHOME/PYTHONPATH again - xpythonsupport.py does.
set "PYTHONHOME=C:\msys64\mingw64"
set "PYTHONPATH=C:\msys64\mingw64\lib\python3.12;C:\msys64\mingw64\lib\python3.12\site-packages;C:\msys64\mingw64\lib\python3.12\lib-dynload"

kame.exe %*
