#!/bin/sh -e
p=/sw
i=/sw
export PREFIX=$p
. ./environment-helper.sh
CC=gcc-3.3 CXX=g++-3.3 ../configure \
 --prefix=$p --with-qt-dir=$p --with-qt-includes=$p/include/qt --with-extra-libs=$p/lib --with-extra-includes=$p/include --enable-mt --with-pic --enable-rpath --enable-shared=yes --enable-static=no --mandir=$i/share/man --with-xinerama --disable-final --disable-dependency-tracking --enable-debug
