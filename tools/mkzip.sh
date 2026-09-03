#!/bin/bash
# Makes the source release: kame-<version>.zip and .tar.bz2, from a clean copy
# of the tree.  This is the release step now that the RPM packaging is gone --
# it was mkrpm.sh, which built from kame.spec, and the spec was the other place
# the version was written.  One place now: kame.pri.
set -e
version=`grep -m1 '^VERSTR' ./kame.pri | sed -e 's/.*=[[:space:]]*//' -e 's/[[:space:]]*$//'`
if [ -z "$version" ]; then
	echo "no VERSTR in ./kame.pri -- run this from the top of the source tree" >&2
	exit 1
fi
file=kame-$version
logfile=mkzip.log
echo $file

dir=../2.1-backups
mkdir -p $dir/$file
rm -f $logfile
rsync --exclude "linux686" \
	--exclude "kdedarwin" \
	--exclude "macosx" \
	--exclude "FTGL" \
	--exclude "/old" \
	--exclude "*.~*" \
	--exclude "*.user*"\
	 --exclude "*.*~"  \
	--exclude "*.log" \
	 --exclude "*.bin" \
	 --exclude "*.dat" \
	 --exclude "*.bix" \
         --exclude "/states" \
         --exclude "*tlaplus/states" \
         --exclude "*tlaplus/doc" --exclude "*tlaplus/doc_ja" \
         --exclude "*tlaplus/*.cfg" --exclude "*tlaplus/*.sh" \
         --exclude "*tlaplus/*.md" --exclude "*tlaplus/*.html" --exclude "*tlaplus/*.py" \
	 --exclude "attic" \
	 --exclude "*.o" --exclude "*.a" --exclude "*.la"  \
	 --exclude "*.app" \
	 --exclude "tools/uipreview/Makefile" \
	 --exclude "*.cache" --exclude ".*" --exclude "*.log"\
         --exclude "*.pyc" --exclude "*.rej" --exclude "*.orig"\
         --exclude "~$*.docx" \
         --exclude "/Testing" \
	 --exclude ".libs" \
         --exclude "bench_kame*" \
         --exclude "bench_mi*" \
         --exclude "bench_sys*" \
	 --exclude "/html" \
	 --exclude "memory" \
	 --exclude "tla2tools.jar" \
	 --exclude "cds_*/genmc" \
         --exclude "*tests/build*" \
	 --exclude "/build*" \
	 --exclude "CVS" \
	 --exclude "odmrimagingng.*" \
         --exclude "*.bak" --exclude "*.qm" \
         --exclude "doc/manual/media" \
         --exclude "tests/Makefile.dyn" --exclude "tests/Makefile.tx" \
         --exclude "tests/Makefile.asp" --exclude "tests/Makefile.3level_mixed" \
	 . $dir/$file -av --delete
# tests/ is a CMake tree now and has no Makefile, so this stopped the script
# dead under set -e.  It is a leftover from the qmake days; the exclusions
# above already keep object files and build directories out, so it only has to
# run where it still applies.
(cd $dir/$file/tests; [ -f Makefile ] && make clean || true)
(cd $dir; tar jcvf $file.tar.bz2 $file)
(cd $dir; zip -9 -r $file.zip $file)
rm -fR $dir/$file

