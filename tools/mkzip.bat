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
copy ..\..\kame\script\rubylineshell.rb ..\kame-win32\resources\
copy ..\..\kame\script\pythonlineshell.py ..\kame-win32\resources\
copy ..\..\kame\script\notebook\jupyter_notebook_config.py ..\kame-win32\resources\
copy ..\..\kame\script\notebook\notebook_kame_kernel_manager.py ..\kame-win32\resources\
copy ..\..\kame\script\kame_mcp_server.py ..\kame-win32\resources\
copy ..\..\kame\script\kame_pydantic_ai.py ..\kame-win32\resources\
copy ..\..\kame\script\kame_python_api.md ..\kame-win32\resources\
copy ..\..\doc\manual\kame-8-en.md ..\kame-win32\resources\
xcopy /S /I /Y ..\..\doc\manual\media ..\kame-win32\resources\media\
xcopy /S /I /Y ..\..\kame\script\plugin ..\kame-win32\resources\plugin\
remove ..\kame-win32\qtdir.txt
remove ..\kame-win32\kame.log
rmdir /S /Y ..\kame-win32\resources\python*\__pycache__
rmdir /S /Y ..\kame-win32\resources\plugin\skills\kame-measurement\__pycache__
remove ..\kame-win32\.qmake*