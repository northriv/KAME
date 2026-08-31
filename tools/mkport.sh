#!/bin/bash
version=`grep -m1 '^VERSTR' ./kame.pri | sed -e 's/.*=[[:space:]]*//' -e 's/[[:space:]]*$//'`
file=kame-$version
logfile=mkport.log
echo $file

dir=../2.1-backups

./tools/mkzip.sh

tarfile=$dir/${file}.tar.bz2
portfile=/opt/localrepo/science/kame/Portfile
distdir=$dir/distfiles
rmd160=`openssl rmd160 ${tarfile} | sed -e 's/RIPEMD160.*= //'`
sha256=`openssl sha256 ${tarfile} | sed -e 's/SHA256.*= //'`
cat $portfile | sed -e "s/checksums.*/checksums               rmd160 $rmd160 \\\\/" -e "s/                        sha256.*/                        sha256 $sha256/" -e "s/version             .*/version             $version/" > port.tmp
sudo mv port.tmp $portfile
#sudo mkdir -p $distdir
sudo cp -f $tarfile $distdir

cmd="sudo port -v -n -k upgrade --force kame +debug_kame -debug -universal"
echo $cmd
$cmd 2>&1 | tee $logfile
#rm -f /tmp/$file.tar.bz2
