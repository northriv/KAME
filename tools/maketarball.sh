#!/bin/bash
file=$1
mkdir -p /tmp/$file
rsync --exclude "linux686" \
	--exclude "kdedarwin" \
	--exclude "macosx" \
	--exclude "FTGL" \
	--exclude "/old" \
	--exclude "*.~*" \
	 --exclude "*.*~"  \
	 --exclude "*.bin" \
	 --exclude "*.dat" \
	 --exclude "attic" \
	 --exclude "*.o" \
	 --exclude "*.a" \
	 --exclude "*.la"  \
	 --exclude ".libs" \
	 --exclude "/html" \
	 --exclude "CVS" \
	 . /tmp/$file -av --delete
(cd /tmp/$file; cat ChangeLog >> kame.spec)
(cd /tmp; tar jcvf $file.tar.bz2 $file)
rm -fR /tmp/$file
#rpmbuild --rcfile=rpmrc -ts $file.tar.bz2
