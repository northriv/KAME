LIST=`find {kame,modules} -name \*.h -o -name \*.cpp`
if test -n "$LIST"; then
 xgettext -ki18n -kI18N_NOOP -ktr2i18n $LIST -o tools/kame.pot;
fi
LIST=`find ../macosx/{kame,modules} -name \*.h -o -name \*.cpp`
if test -n "$LIST"; then
 xgettext -ki18n -kI18N_NOOP -ktr2i18n $LIST -j -o tools/kame.pot;
fi
msgmerge po/ja.po tools/kame.pot -o tools/ja.po
