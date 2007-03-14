#!/bin/bash
emacs -batch $1 -load `pwd`/tools/indent-cpp.el
rm -f $1.~*
