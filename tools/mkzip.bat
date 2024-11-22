copy *.dll ..\kame-win32\
copy kame.exe ..\kame-win32\
copy kame*.bat ..\kame-win32\
copy kame_*.qm ..\kame-win32\
xcopy /S /Y resources\* ..\kame-win32\resources\
copy coremodules2\*.dll ..\kame-win32\coremodules2\
copy coremodules\*.dll ..\kame-win32\coremodules\
copy modules\*.dll ..\kame-win32\modules\
remove ..\kame-win32\qtdir.txt
remove ..\kame-win32\kame.log
rmdir /S /Y ..\kame-win32\resources\python*\__pycache__
remove ..\kame-win32\.qmake*