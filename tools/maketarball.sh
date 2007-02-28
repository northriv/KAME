#!/bin/bash
kame_date=`date +'%Y%m%d'`
version=2.1.2
file=kame-$version-$kame_date
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
sed -e '/kame-lite/d' -e "s/kame_date/$kame_date/" kame.spec > tmp
mv tmp kame.spec
cat ChangeLog >> kame.spec

cd ..
tar jcvf $file.tar.bz2 $file
ln -sf $file.tar.bz2 kame-$version.tar.bz2
rm -fR $file
#rpmbuild --rcfile=rpmrc -ts $file.tar.bz2
