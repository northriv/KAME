@echo off
rem Locate and run KAME's MCP server (kame_mcp_server.py) -- Windows twin of
rem the POSIX launcher next to this file.  Same contract: overrides that are
rem set but wrong FAIL rather than fall through, diagnostics go to stderr
rem (stdout belongs to the MCP stdio stream), and the search order is
rem   KAME_MCP_SERVER > plugin dir > ../ (dev tree) > resource_dir recorded
rem   by KAME in %USERPROFILE%\.kame_kernel_connection.json.
rem Interpreter: KAME_MCP_PYTHON > py -3 > python > python3, first one that
rem imports mcp + jupyter_client.
setlocal enabledelayedexpansion

set "PLUGIN_ROOT=%~dp0.."

rem ---- locate the server script ------------------------------------------
set "SERVER="
if defined KAME_MCP_SERVER (
    if exist "%KAME_MCP_SERVER%" (
        set "SERVER=%KAME_MCP_SERVER%"
    ) else (
        echo kame plugin: KAME_MCP_SERVER is set to "%KAME_MCP_SERVER%" but no such file exists. 1>&2
        exit /b 1
    )
)
if not defined SERVER if exist "%PLUGIN_ROOT%\kame_mcp_server.py" set "SERVER=%PLUGIN_ROOT%\kame_mcp_server.py"
if not defined SERVER if exist "%PLUGIN_ROOT%\..\kame_mcp_server.py" set "SERVER=%PLUGIN_ROOT%\..\kame_mcp_server.py"
if not defined SERVER if exist "%USERPROFILE%\.kame_kernel_connection.json" (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Content -Raw \"$env:USERPROFILE\.kame_kernel_connection.json\" | ConvertFrom-Json).resource_dir" 2^>nul`) do set "RESDIR=%%I"
    if defined RESDIR if exist "!RESDIR!\kame_mcp_server.py" set "SERVER=!RESDIR!\kame_mcp_server.py"
)
if not defined SERVER (
    echo kame plugin: kame_mcp_server.py not found. Set KAME_MCP_SERVER to its full path ^(it is deployed in KAME's Resources directory^). 1>&2
    exit /b 1
)

rem ---- locate an interpreter with the dependencies ------------------------
set "PYEXE="
set "PYARG="
if defined KAME_MCP_PYTHON (
    "%KAME_MCP_PYTHON%" -c "import mcp, jupyter_client" >nul 2>nul
    if errorlevel 1 (
        echo kame plugin: KAME_MCP_PYTHON ^("%KAME_MCP_PYTHON%"^) is not executable or lacks 'mcp' or 'jupyter_client'. 1>&2
        exit /b 1
    )
    set "PYEXE=%KAME_MCP_PYTHON%"
)
if not defined PYEXE call :try "py" "-3"
if not defined PYEXE call :try "python" ""
if not defined PYEXE call :try "python3" ""
if not defined PYEXE (
    echo kame plugin: no Python with 'mcp' and 'jupyter_client' found. Install them ^(pip install mcp jupyter_client^) or set KAME_MCP_PYTHON. 1>&2
    exit /b 1
)

"%PYEXE%" %PYARG% "%SERVER%" --transport=stdio
exit /b %errorlevel%

:try
if defined PYEXE goto :eof
"%~1" %~2 -c "import mcp, jupyter_client" >nul 2>nul || goto :eof
set "PYEXE=%~1"
set "PYARG=%~2"
goto :eof
