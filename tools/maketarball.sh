#!/bin/bash
version=2.2
file=kame-$version
dir=../2.1-backups/$file
mkdir -p $dir
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
	 . $dir -av --delete
cd $dir
#make distclean
mv tmp kame.spec
cat ChangeLog >> kame.spec

cd ..
tar jcvf $file.tar.bz2 $file
rm -fR $file
#rpmbuild --rcfile=rpmrc -ts $file.tar.bz2
