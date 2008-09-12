%global qtver 3.3
%global kdever 3.3
%global ftglver 2.1.2
%global mikachanver 8.9

Name: kame

%{!?build_nidaqmx: %define build_nidaqmx 1}

Version: 2.3.16
Release: 1
License: GPL
Group: Applications/Engineering
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
Requires: qt >= %{qtver}, kdelibs >= %{kdever}
Requires: libart_lgpl, gsl, zlib, ruby, libtool-ltdl, fftw
Requires: libgfortran, atlas-sse2
BuildPreReq: ruby-devel, gsl-devel, boost-devel, libtool-ltdl-devel, fftw-devel
BuildPreReq: libgfortran, atlas-sse2-devel
BuildPreReq: libidn-devel
BuildPreReq: qt-devel >= %{qtver}, kdelibs-devel >= %{kdever}
BuildPreReq: libart_lgpl-devel, zlib-devel, libpng-devel, libjpeg-devel
BuildPreReq: gcc-c++ >= 4.0

Source0: %{name}-%{version}.tar.bz2
Source1: ftgl-%{ftglver}.tar.gz

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
%setup -q -a 1

%build
# build static FTGL
pushd FTGL/unix
CXXFLAGS="-fpermissive -g -O2" ./configure --disable-shared --enable-static
make ##%%{?_smp_mflags}
popd

CXXFLAGS="-g3 -mfpmath=sse -msse -msse2 -mmmx -march=pentium4 -D__sse2__" %configure
make ##%%{?_smp_mflags}

%install
rm -rf $RPM_BUILD_ROOT
%makeinstall
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
%{_datadir}/applications/kde/*.desktop
%{_datadir}/apps/kame
%{_datadir}/icons/*/*/apps/*.png
%{_datadir}/locale/*/LC_MESSAGES/*.mo
%{_datadir}/doc/HTML/*/kame
%{_libdir}/kame/modules/libtestdriver*
%{_libdir}/kame/modules/libmontecarlo*

%files modules-standard
%{_sysconfdir}/udev/rules.d/10-kame.rules
%{_libdir}/kame/libcharinterface*
%{_libdir}/kame/libdsocore*
%{_libdir}/kame/libdmmcore*
%{_libdir}/kame/libmagnetpscore*
%{_libdir}/kame/libdcsourcecore*
%{_libdir}/kame/liblevelmetercore*
%{_libdir}/kame/modules/libdcsource*
%{_libdir}/kame/modules/libdmm*
%{_libdir}/kame/modules/libdso*
%{_libdir}/kame/modules/libfuncsynth*
%{_libdir}/kame/modules/liblia*
%{_libdir}/kame/modules/libmagnetps*
%{_libdir}/kame/modules/libtempcontrol*
%{_libdir}/kame/modules/liblevelmeter*
%{_libdir}/kame/modules/libnetworkanalyzer*

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
