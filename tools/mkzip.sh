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

# Untracked files are never part of a release.  8.6 and 8.6.1 went out with a
# vendor PDF that happened to be sitting in the tree when they were packaged
# -- rsync copies the WORKING tree, and nothing here said otherwise.  git
# knows exactly which files are not the project's, ignored ones included, and
# rsync anchors a pattern to the transfer root with a leading slash.
#
# The working tree stays the source rather than `git archive HEAD`, so that a
# release built with the version bump not yet committed still packages it.
untracked=`mktemp /tmp/mkzip-untracked.XXXXXX`
trap 'rm -f "$untracked"' EXIT
git ls-files --others | sed -e 's|^|/|' > "$untracked"

dir=../2.1-backups
mkdir -p $dir/$file
rm -f $logfile
rsync --exclude-from="$untracked" \
	--exclude "linux686" \
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
# Removed first, because `zip` UPDATES an archive that is already there rather
# than replacing it: a file packaged by an earlier run survives into the next
# one even after it is gone from the tree.  Measured -- a stray PDF deleted
# from the working tree was still in the zip a fresh run produced.  (tar
# truncates on its own; it is here so the pair cannot drift.)
rm -f $dir/$file.tar.bz2 $dir/$file.zip
(cd $dir; tar jcvf $file.tar.bz2 $file)
(cd $dir; zip -9 -r $file.zip $file)
rm -fR $dir/$file

