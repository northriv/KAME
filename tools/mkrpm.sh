#!/bin/bash
make -f admin/Makefile.common cvs
version=`grep Version ./kame.spec | sed -e 's/Version: //'`
file=kame-$version
echo $file
./tools/maketarball.sh $file
export PATH=/usr/libexec/ccache:/usr/lib/ccache:$PATH 
cmd="rpmbuild $1 --target=i686 -ta ../2.1-backups/$file.tar.bz2"
echo $cmd
`$cmd` 2>&1 | tee mkrpm.log
#rm -f /tmp/$file.tar.bz2
