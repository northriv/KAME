#!/bin/bash
dir=../2.1-backups
file=$1
mkdir -p $dir/$file
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
	 --exclude "*.o" --exclude "*.a" --exclude "*.la"  \
	 --exclude "*.cache" --exclude ".*"\
	 --exclude ".libs" \
	 --exclude "/html" \
	 --exclude "CVS" \
	 . $dir/$file -av --delete
(cd $dir/$file/tools/tests; make clean)
(cd $dir/$file; cat ChangeLog >> kame.spec)
(cd $dir; tar jcvf $file.tar.bz2 $file)
rm -fR $dir/$file
#rpmbuild --rcfile=rpmrc -ts $file.tar.bz2
