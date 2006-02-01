#!/bin/sh

if [ -z "$PREFIX" ]; then
	echo "ERROR: set \$PREFIX before sourcing this script!"
	exit 1
fi

if [ `uname -r | cut -d. -f1` = "6" ]; then
	export MACOSX_DEPLOYMENT_TARGET=10.2
	FREETYPE_CONFIG=$PREFIX/bin/freetype-config
else
	export MACOSX_DEPLOYMENT_TARGET=10.3
	export LD_TWOLEVEL_NAMESPACE=true
	FREETYPE_CONFIG=/usr/X11R6/bin/freetype-config
fi

LDFLAGS=""

#for dir in freetype2 xft2 render1 fontconfig2 xrender1 flex; do
for dir in flex; do
	if [ -d "$PREFIX/lib/$dir" ]; then
		if [ -d "$PREFIX/lib/$dir/include/freetype2" ]; then
			export CPPFLAGS="$CPPFLAGS -I$PREFIX/lib/$dir/include/freetype2"
		fi
		export CPPFLAGS="$CPPFLAGS -I$PREFIX/lib/$dir/include"
		export LDFLAGS="-L$PREFIX/lib/$dir/lib -L/usr/X11R6/lib $LDFLAGS"
		export PATH="$PREFIX/lib/$dir/bin:$PATH"
		export PKG_CONFIG_PATH="$PREFIX/lib/$dir/lib/pkgconfig"
	fi
done

# -fast, minus the unsafe bits
if [ -d "libltdl" ]; then
	export ACLOCALFLAGS="$ACLOCALFLAGS -I libltdl"
fi
export CFLAGS="-O2 -g -fPIC"
export CXXFLAGS="$CFLAGS"
export CPPFLAGS="$CPPFLAGS -fno-common -no-cpp-precomp -DMACOSX -DARTS_NO_ALARM -I$PREFIX/include -I/usr/X11R6/include"
export FREETYPE_CONFIG
export LIBS="$LIBS -L$PREFIX/lib -L/usr/X11R6/lib"
export SED="sed"

PATH=`echo $PATH | perl -p -e 'for my $entry (split(/:/, $_)) { next if ($entry =~ m,^/usr/local,); push(@path, $entry) }; $_ = join(":", @path)'`
export PATH="$PATH:/usr/X11R6/bin"

export ALL_LIBRARIES="$LDFLAGS -L$PREFIX/lib -L/usr/X11R6/lib"

cat <<END
the following environment is being used:

  ACLOCALFLAGS:    $ACLOCALFLAGS
  CFLAGS:          $CFLAGS
  CPPFLAGS:        $CPPFLAGS
  CXXFLAGS:        $CXXFLAGS
  FREETYPE_CONFIG: $FREETYPE_CONFIG
  LDFLAGS:         $LDFLAGS
  LIBS:            $LIBS
  PATH:            $PATH
  PKG_CONFIG_PATH: $PKG_CONFIG_PATH

END

