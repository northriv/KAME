copy *.dll ..\kame-win32\
copy kame.exe ..\kame-win32\
copy kame*.bat ..\kame-win32\
copy kame_*.qm ..\kame-win32\
xcopy /S /Y resources\* ..\kame-win32\resources\
copy coremodules2\*.dll ..\kame-win32\coremodules2\
copy coremodules\*.dll ..\kame-win32\coremodules\
copy modules\*.dll ..\kame-win32\modules\
rem --- AI support (MCP server + API ref + Pydantic AI + user manual + plugin).
rem     Windows qmake does NOT deploy scriptfile.files (they are in DISTFILES
rem     only), so the build tree's resources\ never receives them -- copy them
rem     straight from the source tree instead, mirroring the macOS bundle's
rem     Contents/Resources layout.  Without kame_mcp_server.py the release has
rem     no MCP server at all; kame_python_api.md / kame-8-en.md + media\ back
rem     the kame_api and kame_manual tools.  plugin\ is for macOS parity -- it
rem     is inert on Windows (its .mcp.json invokes a POSIX-sh launcher).
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