#!/bin/bash
for i in `find kame -name '*.cpp' -or -name '*.h'\
 -and ! -name 'atomic_prv*.h' -and ! -name 'threadlocal.h' \
 -and ! -name '*icon.cpp' -and ! -name '*userdmm.h'`
do
	echo $i
	`pwd`/tools/indent.sh $i
done
