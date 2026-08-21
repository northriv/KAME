@echo off
rem Deploy kame.pro's scriptfile.files into a build/release resources dir.
rem
rem Windows qmake deploys none of them (they are DISTFILES only), unlike the
rem macOS bundle and the Linux QMAKE_POST_LINK, so kame.exe used to start with
rem no kame_mcp_server.py next to it and the MCP link died with
rem "can't open file ...\Resources\kame_mcp_server.py".  Called from
rem kame/kame.pro's win32 QMAKE_POST_LINK, and by tools/mkzip.bat.
rem
rem Usage: deploy_scripts.bat <dest-resources-dir>
setlocal
if "%~1"=="" (echo usage: %~nx0 ^<dest-resources-dir^> & exit /b 2)
set DEST=%~1
rem This script lives in <src>\tools, so the source root is one level up.
set SRC=%~dp0..

if not exist "%DEST%" mkdir "%DEST%"

for %%f in (
  "script\rubylineshell.rb"
  "script\pythonlineshell.py"
  "script\notebook\jupyter_notebook_config.py"
  "script\notebook\notebook_kame_kernel_manager.py"
  "script\kame_mcp_server.py"
  "script\kame_pydantic_ai.py"
  "script\kame_python_api.md"
) do (
  if exist "%SRC%\kame\%%~f" (
    copy /Y "%SRC%\kame\%%~f" "%DEST%\" >nul || echo   WARN: failed %%~f
  ) else (
    echo   WARN: missing %SRC%\kame\%%~f
  )
)

if exist "%SRC%\doc\manual\kame-8-en.md" copy /Y "%SRC%\doc\manual\kame-8-en.md" "%DEST%\" >nul
if exist "%SRC%\doc\manual\media" xcopy /S /I /Y /Q "%SRC%\doc\manual\media" "%DEST%\media\" >nul
if exist "%SRC%\kame\script\plugin" xcopy /S /I /Y /Q "%SRC%\kame\script\plugin" "%DEST%\plugin\" >nul

rem __pycache__ is nested; a single rmdir would leave the inner ones behind.
for /d /r "%DEST%" %%d in (__pycache__) do @if exist "%%d" rd /S /Q "%%d"

echo   deployed script files to %DEST%
endlocal
exit /b 0
