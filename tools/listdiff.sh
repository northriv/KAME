#!/bin/bash
for against in 1.8.2-20041210
do
file=kame-1.8.2-`date +'%Y%m%d%H%M'`-against-$against.diff
path=../1.8-backups/$file
dir=../1.8-backups/kame-$against/kame
diff  -ur -x '*.[ao]' -x '.*' -x '*.lo' -x '*.la' -x '*.ui' -x 'forms' -x 'Makefile*' -x 'config*' -x 'autom*' -x '*.moc*' -x '*.txt' -x '*.*~' $dir kame > $path
ln -sf $path difflist.txt
done
