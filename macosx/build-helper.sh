#!/bin/sh

LOGDIR="/tmp/buildlog"
mkdir -p "$LOGDIR"

TYPE="$1";     shift
NAME="$1";     shift
VERSION="$1";  shift
REVISION="$1"; shift

rm -rf "$LOGDIR/$NAME-$VERSION-$REVISION.failed"

date > "$LOGDIR/$NAME-$VERSION-$REVISION.$TYPE"
("$@" || touch "$LOGDIR/$NAME-$VERSION-$REVISION.failed") 2>&1 | tee -a "$LOGDIR/$NAME-$VERSION-$REVISION.$TYPE"

if [ -f "$LOGDIR/$NAME-$VERSION-$REVISION.failed" ]; then
	echo "NAME-$VERSION-$REVISION $TYPE-build failed!"
	rm -rf "$LOGDIR/$NAME-$VERSION-$REVISION.failed"
	exit 1
fi
