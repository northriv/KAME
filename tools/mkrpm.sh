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
	 --exclude "*.*~"  \
	 --exclude "*.bin" \
	 --exclude "*.dat" \
	 --exclude "attic" \
	 --exclude "*.o" --exclude "*.a" --exclude "*.la"  \
	 --exclude "*.cache" --exclude ".*" --exclude "*.log"\
	 --exclude ".libs" \
	 --exclude "/html" \
	 --exclude "CVS" \
	 . $dir/$file -av --delete
(cd $dir/$file/tools/tests; make clean)
(cd $dir/$file; cat ChangeLog >> kame.spec)
(cd $dir; tar jcvf $file.tar.bz2 $file)
rm -fR $dir/$file

export PATH=/usr/libexec/ccache:/usr/lib/ccache:$PATH 
cmd="rpmbuild $1 --target=i686 -ta ../2.1-backups/$file.tar.bz2"
echo $cmd
`$cmd` 2>&1 | tee $logfile
#rm -f /tmp/$file.tar.bz2
