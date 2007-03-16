#!/bin/bash
version=$1
file=kame-$version
./tools/maketarball $file
target=${2:-i686}
echo TARGET=$target
export PATH=/usr/libexec/ccache:/usr/lib/ccache:$PATH 
rpmbuild --target $target -ta /tmp/$file.tar.bz2 2>&1 | tee /tmp/mkrpm.log
#rm -f /tmp/$file.tar.bz2
