copy *.dll ..\kame-win32\
copy kame.exe ..\kame-win32\
copy kame*.bat ..\kame-win32\
copy kame_*.qm ..\kame-win32\
xcopy /S /Y resources\* ..\kame-win32\resources\
copy coremodules2\*.dll ..\kame-win32\coremodules2\
copy coremodules\*.dll ..\kame-win32\coremodules\
copy modules\*.dll ..\kame-win32\modules\
rem --- kame.pro's scriptfile.files, copied from the SOURCE tree.
rem     Windows qmake never deploys them (they sit in DISTFILES only), so the
rem     build tree's resources\ only ever holds whatever was hand-copied there
rem     once -- which is how a release shipped a jupyter_notebook_config.py and
rem     notebook_kame_kernel_manager.py predating the orphan-server watchdog and
rem     the working interrupt/restart overrides, and no kame_mcp_server.py at
rem     all.  Copying the whole set from source every time keeps that from
rem     silently rotting again; it mirrors the macOS Contents/Resources layout.
rem     kame_python_api.md / kame-8-en.md + media\ back the kame_api and
rem     kame_manual MCP tools.  plugin\ is for macOS parity -- it is inert on
rem     Windows (its .mcp.json invokes a POSIX-sh launcher).
rem     Shared with kame.pro's win32 QMAKE_POST_LINK, so a build tree and a
rem     release get exactly the same set.
call ..\..\tools\deploy_scripts.bat ..\kame-win32\resources
del /Q ..\kame-win32\qtdir.txt 2>nul
del /Q ..\kame-win32\kame.log 2>nul
del /Q ..\kame-win32\.qmake* 2>nul
rem __pycache__ is NESTED (python3.12\collections\, importlib\metadata\, ...),
rem so a single `rmdir resources\python*\__pycache__` cleared only the top
rem level and shipped the rest.  `for /d /r` walks every subdirectory.
rem (`remove` used to be here, but that is not a cmd command -- these deletes
rem  never ran, which is why qtdir.txt and kame.log kept reaching releases.)
for /d /r "..\kame-win32" %%d in (__pycache__) do @if exist "%%d" rd /S /Q "%%d"