#!/bin/bash
version=`grep Version ./kame.spec | sed -e 's/Version: //'`
file=kame-$version
logfile=mkrpm.log
echo $file

dir=../2.1-backups
mkdir -p $dir/$file
rm $logfile
rsync --exclude "linux686" \
	--exclude "kdedarwin" \
	--exclude "macosx" \
	--exclude "FTGL" \
	--exclude "/old" \
	--exclude "*.~*" \
	--exclude "*.user"\
	 --exclude "*.*~"  \
	--exclude "*.log" \
	 --exclude "*.bin" \
	 --exclude "*.dat" \
	 --exclude "attic" \
	 --exclude "*.o" --exclude "*.a" --exclude "*.la"  \
	 --exclude "*.cache" --exclude ".*" --exclude "*.log"\
	 --exclude ".libs" \
	 --exclude "/html" \
	 --exclude "CVS" \
	 --exclude "odmrimagingng.*" \
	 . $dir/$file -av --delete
(cd $dir/$file/tests; make clean)
(cd $dir/$file; cat ChangeLog >> kame.spec)
(cd $dir; tar jcvf $file.tar.bz2 $file)
(cd $dir; zip -9 -r $file.zip $file)
rm -fR $dir/$file

