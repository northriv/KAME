#!/bin/bash
make -f admin/Makefile.common cvs
version=2.2.3
file=kame-$version
./tools/maketarball.sh $file
target=${2:-i686}
echo TARGET=$target
export PATH=/usr/libexec/ccache:/usr/lib/ccache:$PATH 
rpmbuild --target $target -ta ../2.1-backups/$file.tar.bz2 2>&1 | tee mkrpm.log
#rm -f /tmp/$file.tar.bz2
