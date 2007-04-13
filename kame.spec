%global qtver 3.3
%global kdever 3.3
%global ftglver 2.1.2
%global mikachanver 8.9

Name: kame

%{!?build_nidaqmx: %define build_nidaqmx 1}

Version: 2.2.3
Release: 1
License: GPL
Group: Applications/Engineering
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
Requires: qt >= %{qtver}, kdelibs >= %{kdever}
Requires: libart_lgpl, gsl, zlib, ruby, libtool-ltdl
BuildPreReq: ruby-devel, gsl-devel, boost-devel, libtool-ltdl-devel
BuildPreReq: libidn-devel
BuildPreReq: qt-devel >= %{qtver}, kdelibs-devel >= %{kdever}
BuildPreReq: libart_lgpl-devel, zlib-devel, libpng-devel, libjpeg-devel
BuildPreReq: gcc-c++ >= 3.3
BuildPreReq: compat-gcc-34-c++

Source0: %{name}-%{version}.tar.bz2
Source1: ftgl-%{ftglver}.tar.gz
Source2: mikachanfont-%{mikachanver}.tar.bz2

Summary: KAME, K's adaptive measurement engine.

%description
K's adaptive measurement engine. 

%package modules-standard
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame = %{version}
Requires: linux-gpib
BuildPreReq: linux-gpib-devel
%description modules-standard
K's adaptive measurement engine.
Many standard drivers.

%package modules-nmr
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame-modules-standard = %{version}
Requires: fftw2
BuildPreReq: fftw2-devel
%description modules-nmr
K's adaptive measurement engine.
NMR drivers.

%if 0%{build_nidaqmx}
%package modules-nidaq
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame-modules-nmr = %{version}
Requires: nidaqmxef
BuildPreReq: nidaqmxcapii
%description modules-nidaq
K's adaptive measurement engine.
NMR drivers.
%endif

%prep
%setup -q -a 1 -a 2
mv mikachanfont-%{mikachanver}/fonts/* kame/mikachanfont
mv mikachanfont-%{mikachanver}/* kame/mikachanfont

%build
# build static FTGL
pushd FTGL/unix
CXX=g++34 ./configure --disable-shared --enable-static
make ##%%{?_smp_mflags}
popd

%configure --enable-debug
make ##%%{?_smp_mflags}

%install
rm -rf $RPM_BUILD_ROOT
%makeinstall
#if [ -f $RPM_BUILD_ROOT/%{_bindir}/*-kame ]
#then
#	mv $RPM_BUILD_ROOT/%{_bindir}/*-kame $RPM_BUILD_ROOT/%{_bindir}/kame
#fi

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%{_bindir}/kame
%{_datadir}/applnk/Applications/*.desktop
%{_datadir}/apps/kame
%{_datadir}/icons/*/*/apps/*.png
%{_datadir}/locale/*/LC_MESSAGES/*.mo
%{_datadir}/doc/HTML/*/kame
%{_libdir}/kame/modules/libtestdriver*
%{_libdir}/kame/modules/libmontecarlo*

%files modules-standard
%{_libdir}/kame/libcharinterface*
%{_libdir}/kame/libdsocore*
%{_libdir}/kame/libdmmcore*
%{_libdir}/kame/libmagnetpscore*
%{_libdir}/kame/libdcsourcecore*
%{_libdir}/kame/modules/libdcsource*
%{_libdir}/kame/modules/libdmm*
%{_libdir}/kame/modules/libdso*
%{_libdir}/kame/modules/libfuncsynth*
%{_libdir}/kame/modules/liblia*
%{_libdir}/kame/modules/libmagnetps*
%{_libdir}/kame/modules/libtempcontrol*

%files modules-nmr
%{_libdir}/kame/libnmrpulsercore*
%{_libdir}/kame/libsgcore*
%{_libdir}/kame/modules/libnmr*
%{_libdir}/kame/modules/libsg*

%if 0%{build_nidaqmx}
%files modules-nidaq
%{_libdir}/kame/modules/libnidaq*
%endif

%changelog
