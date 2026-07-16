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
(cd $dir/$file/tests; make clean)
(cd $dir/$file; cat ChangeLog >> kame.spec)
(cd $dir; tar jcvf $file.tar.bz2 $file)
(cd $dir; zip -9 -r $file.zip $file)
rm -fR $dir/$file

