#!/bin/bash
for i in `find kame -name '*.cpp' -or -name '*.h'`
do
echo $i
emacs -batch $i -load `pwd`/tools/indent-cpp.el
rm -f $i.~*
done
