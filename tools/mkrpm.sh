#!/bin/bash
version=`grep Version ./kame.spec | sed -e 's/Version: //'`
file=kame-$version
logfile=mkrpm.log
echo $file

dir=../2.1-backups

./tools/maketarball.sh

tarfile=$dir/${file}.tar.bz2

export PATH=/usr/libexec/ccache:/usr/lib/ccache:$PATH 
cmd="rpmbuild $1 --target=i686 -ta $tarfile"
echo $cmd
$cmd 2>&1 | tee $logfile
#rm -f /tmp/$file.tar.bz2
