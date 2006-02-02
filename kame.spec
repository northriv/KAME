%define momorel 1
%global kamedate kame_date
%global qtver 3.3
%global kdever 3.3
%global ftglver 2.1.2
%global mikachanver 8.9

Name: kame
Version: 2.0.1
Release: %{momorel}m
License: GPL
Group: Applications/Engineering
BuildRoot: %{_tmppath}/%{name}-%{version}-root
Requires: qt >= %{qtver}, kdelibs >= %{kdever}
Requires: XFree86, libart_lgpl, gsl, fftw, zlib, freetype2
BuildPreReq: ruby-devel, gsl, boost-devel, fftw-devel
BuildPreReq: libidn-devel, freetype2-devel
BuildPreReq: qt-devel >= %{qtver}, kdelibs-devel >= %{kdever}
BuildPreReq: XFree86-devel, libart_lgpl-devel, zlib-devel, libpng-devel, libjpeg-devel
BuildPreReq: gcc-c++ >= 3.3
Source0: %{name}-%{version}-%{kamedate}.tar.bz2
Source1: ftgl-%{ftglver}.tar.gz
Source2: mikachanfont-%{mikachanver}.tar.bz2

Summary: KAME, K's adaptive measurement engine.

%description
K's adaptive measurement engine. 
kame-lite is a NMR disabled version of kame.

%prep
%setup -q -a 1 -a 2 -n %{name}-%{version}-%{kamedate}
mv mikachanfont-%{mikachanver}/fonts/* kame/mikachanfont
mv mikachanfont-%{mikachanver}/* kame/mikachanfont

%build
# build static FTGL
pushd FTGL/unix
./configure --disable-shared --enable-static
make
popd

%configure --enable-debug
%make %{?_smp_mflags}

%install
rm -rf %{buildroot}
%makeinstall
if [ -f %{buildroot}/%{_bindir}/*-kame ]
then
	mv %{buildroot}/%{_bindir}/*-kame %{buildroot}/%{_bindir}/kame
fi

%clean
rm -rf %{buildroot}

%files
%defattr(-,root,root)
%{_bindir}/kame
%{_datadir}/applnk/Applications/*.desktop
%{_datadir}/apps/kame
%{_datadir}/icons/*/*/apps/*.png
%{_datadir}/locale/*/LC_MESSAGES/*.mo
%{_datadir}/doc/HTML/*/kame

%changelog
