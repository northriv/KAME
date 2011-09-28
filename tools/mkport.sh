#!/bin/bash
version=`grep Version ./kame.spec | sed -e 's/Version: //'`
file=kame-$version
logfile=mkrpm.log
echo $file

dir=../2.1-backups

./tools/maketarball.sh

tarfile=$dir/${file}.tar.bz2
portfile=../../macports/kde/kame/Portfile
distdir=$dir/distfiles
md5=`md5 ${tarfile} | sed -e 's/MD5.*= //'`
cat $portfile | sed -e "s/checksums.*/checksums           md5 $md5/" -e "s/version             .*/version             $version/" > port.tmp && mv port.tmp $portfile
#sudo mkdir -p $distdir
sudo cp -f $tarfile $distdir

cmd="sudo port -v -n -k -u upgrade kame +debug_kame -debug -universal"
echo $cmd
$cmd 2>&1 | tee $logfile
#rm -f /tmp/$file.tar.bz2
