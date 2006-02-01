%define	fontdir	%{_datadir}/fonts/ja/TrueType
%define	fontname mikachan

Summary:	Japanese TrueType fonts made by mikachan
Name:		mikachanfont
Version:	8.8
Release:	1
License:	Distributable
Group:		User Interface/X
BuildArch:	noarch
BuildRoot:	%{_tmppath}/%{name}-%{PACKAGE_VERSION}-root
Source:		%{name}-%{version}.tar.bz2
URL:		http://www02.u-page.so-net.ne.jp/dc4/hiro1/mika/index.html
BuildRequires:	XFree86-xfs, VFlib2
PreReq:		/usr/sbin/chkfontpath /usr/X11R6/bin/mkfontdir chkfontscale >= 0.1

%description
This package provides Japanese TrueType fonts, and includes
ttindex files for VFlib.

%prep
%setup -q

%build
cd fonts
ttindex %{fontname}

%install
[ -n "$RPM_BUILD_ROOT" -a "$RPM_BUILD_ROOT" != / ] && rm -rf $RPM_BUILD_ROOT

install -d $RPM_BUILD_ROOT%{fontdir}
install -m 0644 fonts/*.{ttf,tti} $RPM_BUILD_ROOT%{fontdir}
install -m 0644 fontsconf/* $RPM_BUILD_ROOT%{fontdir}

%clean
[ -n "$RPM_BUILD_ROOT" -a "$RPM_BUILD_ROOT" != / ] && rm -rf $RPM_BUILD_ROOT

%post
fc-cache
/usr/X11R6/bin/chkfontscale -a %{fontdir}/fonts.scale.%{name} %{fontdir}
/usr/X11R6/bin/mkfontdir -e /usr/X11R6/lib/X11/fonts/encodings \
                         -e /usr/X11R6/lib/X11/fonts/encodings\large \
                          %{fontdir}
/usr/sbin/chkfontpath -q -a %{fontdir}
if [ -x /usr/bin/redhat-update-gnome-font-install ]; then
	/usr/bin/redhat-update-gnome-font-install
fi
if [ -x /usr/bin/redhat-update-gnome-font-install2 ]; then
	/usr/bin/redhat-update-gnome-font-install2
fi

%postun
/usr/X11R6/bin/chkfontscale -r %{fontname}.ttf %{fontdir}
/usr/X11R6/bin/mkfontdir -e /usr/X11R6/lib/X11/fonts/encodings \
                         -e /usr/X11R6/lib/X11/fonts/encodings\large \
                          %{fontdir}
if [ "$1" = "0" ];
then
	/usr/sbin/chkfontpath -q -r %{fontdir}
	if [ -x /usr/bin/redhat-update-gnome-font-install ]; then
		/usr/bin/redhat-update-gnome-font-install
	fi
	if [ -x /usr/bin/redhat-update-gnome-font-install2 ]; then
		/usr/bin/redhat-update-gnome-font-install2
	fi
fi

%files
%defattr(-,root,root)
%doc README README.ja COPYRIGHT COPYRIGHT.ja ChangeLog
%{fontdir}/%{fontname}.ttf
%{fontdir}/%{fontname}.tti
%{fontdir}/fonts.scale.%{name}

%changelog
* Sun Feb 16 2003 Yuhei Matsunaga <yuhei@users.sourceforge.jp> 8.8-1
- Version up.
* Tue Dec 17 2002 Yuhei Matsunaga <yuhei@users.sourceforge.jp> 8.7-1
- Version up.
* Mon Dec 9 2002 Yuhei Matsunaga <yuhei@users.sourceforge.jp> 8.6-1
- Initial build.
