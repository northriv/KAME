%global qtver 4.4
%global kdever 4.3

Name: kame

%{!?build_nidaqmx: %define build_nidaqmx 1}

Version: 4.1.14
Release: 1
License: GPL
Group: Applications/Engineering
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
Requires: qt >= %{qtver}, kdelibs >= %{kdever}
Requires: libart_lgpl, gsl, zlib, ruby, libtool-ltdl, fftw
Requires: libgfortran, atlas-sse2
Requires: oxygen-icon-theme
BuildRequires: ruby, ruby-devel, gsl-devel, boost-devel, libtool, libtool-ltdl-devel, fftw-devel
BuildRequires: gcc-gfortran, atlas-sse2-devel
BuildRequires: libidn-devel
BuildRequires: qt-devel >= %{qtver}, kdelibs >= %{kdever}, kdelibs-devel >= %{kdever}
BuildRequires: zlib-devel, libpng-devel, libjpeg-devel
BuildRequires: gcc-c++ >= 4.0
#BuildRequires: clang >= 2.9

Source0: %{name}-%{version}.tar.bz2

Summary: KAME, K's adaptive measurement engine.

%description
K's adaptive measurement engine. 

%package modules-standard
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame = %{version}
Requires: linux-gpib
BuildRequires: linux-gpib-devel
%description modules-standard
K's adaptive measurement engine.
Many standard drivers.

%package modules-nmr
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame-modules-standard = %{version}
%description modules-nmr
K's adaptive measurement engine.
NMR drivers.

%if 0%{build_nidaqmx}
%package modules-nidaq
Group: Applications/Engineering
Summary: KAME, K's adaptive measurement engine. Modules.
Requires: kame-modules-nmr = %{version}
Requires: nidaqmxef
BuildRequires: nidaqmxcapii
%description modules-nidaq
K's adaptive measurement engine.
NMR drivers.
%endif

%prep

%setup -q
export CXX="g++ -g3 -O2 -mfpmath=sse -msse -msse2 -mmmx -march=pentium4 -D__sse2__"
#export CXX="clang++ -g -O4 -march=pentium4 -D__sse2__"
%cmake #-DCMAKE_BUILD_TYPE=Debug


%build
make

%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT
#if [ -f $RPM_BUILD_ROOT/%{_bindir}/*-kame ]
#then
#	mv $RPM_BUILD_ROOT/%{_bindir}/*-kame $RPM_BUILD_ROOT/%{_bindir}/kame
#fi
%if !0%{build_nidaqmx}
	rm -f $RPM_BUILD_ROOT%{_libdir}/kame/modules/libnidaq*
%endif

mkdir -p $RPM_BUILD_ROOT%{_sysconfdir}/udev/rules.d
cat <<EOF > $RPM_BUILD_ROOT%{_sysconfdir}/udev/rules.d/10-kame.rules
KERNEL=="ttyUSB*", GROUP="uucp", MODE="0666"
KERNEL=="ttyS*", GROUP="uucp", MODE="0666"
EOF

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%{_bindir}/kame
%{_datadir}/applications/kde4/*.desktop
%{_datadir}/kde4/apps/kame
%{_datadir}/icons/*/*/apps/*.png
%{_datadir}/locale/*/LC_MESSAGES/*.mo
%{_datadir}/doc/HTML/*/kame
%{_libdir}/kame/modules/libtestdriver*
%{_libdir}/kame/modules/libmontecarlo*

%files modules-standard
%{_sysconfdir}/udev/rules.d/10-kame.rules
%{_libdir}/libkame_charinterface*
%{_libdir}/libkame_dcsourcecore*
%{_libdir}/libkame_dmmcore*
%{_libdir}/libkame_dsocore*
%{_libdir}/libkame_magnetpscore*
%{_libdir}/libkame_levelmetercore*
%{_libdir}/libkame_networkanalyzercore*
%{_libdir}/libkame_motorcore*
%{_libdir}/libkame_flowcontrollercore*
%{_libdir}/kame/modules/libcounter*
%{_libdir}/kame/modules/libdcsource*
%{_libdir}/kame/modules/libdmm*
%{_libdir}/kame/modules/libdso*
%{_libdir}/kame/modules/libfuncsynth*
%{_libdir}/kame/modules/liblia*
%{_libdir}/kame/modules/libmagnetps*
%{_libdir}/kame/modules/libtempcontrol*
%{_libdir}/kame/modules/liblevelmeter*
%{_libdir}/kame/modules/libnetworkanalyzer*
%{_libdir}/kame/modules/libfourres*
%{_libdir}/kame/modules/libmotor*
%{_libdir}/kame/modules/libflowcontroller*
%{_libdir}/kame/modules/libqd*

%files modules-nmr
%{_libdir}/libkame_nmrpulsercore*
%{_libdir}/libkame_sgcore*
%{_libdir}/kame/modules/libnmr*
%{_libdir}/kame/modules/libsg*
%{_libdir}/kame/modules/libthamway*

%if 0%{build_nidaqmx}
%files modules-nidaq
%{_libdir}/kame/modules/libnidaq*
%endif

#the following will be copied from the separated file ChangeLog.
%changelog
