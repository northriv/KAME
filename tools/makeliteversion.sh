#!/bin/bash
kame_date=`date +'%Y%m%d'`
file=kame-lite-2.0-$kame_date
dir=../2.0-backups/$file
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
	 --exclude "/tools" \
	 --exclude "/kame/users/nmr" \
	 . $dir -av --delete
cd $dir
#make distclean
sed -e '/nmr/d' kame/Makefile.am > tmp
mv tmp kame/Makefile.am
sed -e '/nmr/d' kame/users/userdrivers.cpp > tmp
mv tmp kame/users/userdrivers.cpp
sed -e 's/nmr//' kame/users/Makefile.am > tmp
mv tmp kame/users/Makefile.am
sed -e 's/Name: kame/Name: kame-lite/' -e "s/kame_date/$kame_date/" kame.spec > tmp
mv tmp kame.spec
cat ChangeLog >> kame.spec

cd ..
tar jcvf $file.tar.bz2 $file
ln -sf $file.tar.bz2 kame-lite-2.0.tar.bz2
#rm -fR $file
#rpmbuild --rcfile=rpmrc -ts $file.tar.bz2
