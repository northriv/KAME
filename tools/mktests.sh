#!/bin/bash
dir=../2.1-backups/stm_tests
mkdir -p $dir
rsync -auv COPYING AUTHORS tests/[astx]*.cpp tests/support.h kame/{allocator,thread}.cpp kame/{allocator,atomic*,thread*,pthread*,transaction*,xtime,xscheduler,xsignal*}.h $dir
rsync -auv tests/Makefile.tests $dir/Makefile

tarfile=$dir/../stmtests.tar.gz
(cd $dir; make clean)
(cd $dir/..; tar cvzf $tarfile stm_tests)
#(cd $dir; make check)
