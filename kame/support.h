/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//Every KAME source must include this header
//---------------------------------------------------------------------------

#ifndef supportH
#define supportH

#define quotedefined(str) #str

#define KAME_DATAFILE_DELIMITER " " //Space

#ifndef DECLSPEC_KAME
    #define DECLSPEC_KAME
#endif
#ifndef DECLSPEC_MODULE
    #define DECLSPEC_MODULE
#endif
#ifndef DECLSPEC_SHARED
    #define DECLSPEC_SHARED
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// Byte order.  `__BIG_ENDIAN__` is predefined by clang/Apple GCC only; plain
// GCC never defines it, and the only other thing that used to (autoconf's
// WORDS_BIGENDIAN via config.h) went away with the autotools build.  So on
// Linux/GCC every `#ifdef __BIG_ENDIAN__` in the raw-stream layer
// (kame/driver/primarydriver.h push_*/pop_*, whose documented file format is
// little-endian, and the camera mono16 swap in digitalcamera.cpp) was dead
// code with no branch covering the platform at all — correct by accident on
// x86-64/aarch64, silently wrong on s390x or ppc64be.  Derive it from the
// compiler's own macros, which GCC, clang and MSVC (little-endian only) all
// provide, and keep WORDS_BIGENDIAN honoured for anyone still passing it.
#if defined WORDS_BIGENDIAN || \
    (defined __BYTE_ORDER__ && defined __ORDER_BIG_ENDIAN__ && \
     __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    #ifndef __BIG_ENDIAN__
        #define __BIG_ENDIAN__
    #endif
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#if !defined M_PI
    #define M_PI 3.1415926535897932385
#endif

#include <cstdint>   //!< Fixed-width integer types are used all over the
                     //!< tree.  They used to arrive only by leaking out of
                     //!< <string>, which newer libstdc++ no longer does —
                     //!< guarantee them here instead of in every header.
#include <cassert>
#ifdef NDEBUG
#define DEBUG_XTHREAD 0
#else
#define DEBUG_XTHREAD 1
#endif

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #define USE_QTHREAD
    #include <QThread>
    #include <thread>
    // Windows custom mlock (defined in support.cpp via VirtualLock).
    DECLSPEC_KAME int mlock(const void *addr, size_t len);
#else
    #include <pthread.h>
    #define USE_PTHREAD
    // POSIX mlock — declared by <sys/mman.h>. Pulled in here so any
    // translation unit that includes "support.h" sees it (formerly
    // came in transitively via xthread.h).
    #include <sys/mman.h>
#endif

#include <memory>
using std::unique_ptr;
using std::shared_ptr;
using std::weak_ptr;
using std::enable_shared_from_this;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;

#include <stdio.h>
#include <algorithm>
#include <string>
#include <QString>

#if defined(WITH_KDE)
    #include <klocale.h>
    #define i18n_noncontext(src) i18n(src)
#else
    #include <QCoreApplication>
    #define i18n_noncontext(src) QCoreApplication::translate("static", src)
    #include <type_traits>
    #define i18n(src) ((std::is_base_of<QObject, decltype( *this)>::value) ?\
        QObject::tr(src) : i18n_noncontext(src))
    #define I18N_NOOP(txt) QT_TR_NOOP(txt)
#endif

class DECLSPEC_KAME XString : public std::string {
using base_type = std::string;
public:
    XString() = default;
    XString(const XString&) = default;
    XString(XString&&) noexcept = default;
    XString(const char *str) : base_type(str) {}
    XString(const QString &str) : base_type() {
        const auto &s = str.toUtf8();
        base_type x(s.constData());
        x.swap( *this);
    }
    XString(const base_type &str) : base_type(str) {}
    operator QString() const {return QString::fromUtf8(c_str());}
    XString operator+(const char *s) {return *this + base_type(s);}
    XString &operator=(const XString&) = default;
    XString &operator=(XString&&) noexcept = default;
};

//! Debug printing.
#define dbgPrint(msg) dbgPrint_redirected(msg, __FILE__, __LINE__, false)
#define gMessagePrint(msg) dbgPrint_redirected(msg, __FILE__, __LINE__, true)
DECLSPEC_KAME void
dbgPrint_redirected(const XString &str, const char *file, int line, bool force_dump);
//! Global Error Message/Printing.
#define gErrPrint(msg) gErrPrint_redirected(msg, __FILE__, __LINE__)
#define gWarnPrint(msg) gWarnPrint_redirected(msg, __FILE__, __LINE__)
DECLSPEC_KAME void
gErrPrint_redirected(const XString &str, const char *file, int line);
DECLSPEC_KAME void
gWarnPrint_redirected(const XString &str, const char *file, int line);

#include <stdexcept>
//! Base of exception
struct DECLSPEC_KAME XKameError : public std::runtime_error {
	XKameError();
    virtual ~XKameError() = default;

	//! errno is read and cleared after a construction
	XKameError(const XString &s, const char *file, int line);
    void print();
	void print(const XString &header);
	static void print(const XString &msg, const char *file, int line, int errno_);
	const XString &msg() const;
    virtual const char* what() const noexcept;
private:
	XString m_msg;
	const char * m_file;
	int m_line;
	int m_errno;
};

//! If true, Log all dbgPrint().
DECLSPEC_KAME extern bool g_bLogDbgPrint;
//! If true, use mlock.
DECLSPEC_KAME extern bool g_bUseMLock;

DECLSPEC_KAME bool isMemLockAvailable() noexcept;

//! CPU idle-latency (PM-QoS) request, held for the lifetime of the object.
//!
//! On Linux, holding \c /dev/cpu_dma_latency open with a target written into it
//! tells the cpuidle governor to use only C-states whose *exit* latency is at
//! or below that target; the kernel aggregates every open request system-wide
//! and takes the minimum, and the constraint is dropped the moment the file
//! descriptor is closed.  This is the standard way for an acquisition program
//! to buy back deep-C-state wake-up latency for the duration of a measurement
//! without disabling idle states globally (which costs power and, on a
//! thermally tight machine, invites throttling that makes the numbers worse).
//!
//! Typical exit latencies on a Kaby Lake desktop part: C1 2 us, C1E 10 us,
//! C3 70 us, C6 85 us, C7s 124 us, C8 200 us.  A target of 10 leaves POLL, C1
//! and C1E; 0 pins the CPUs awake.
//!
//! This matters when a completion has to reach a sleeping thread promptly --
//! notably when instrument IRQs are steered to housekeeping cores and the
//! acquisition thread lives on an isolated core, so the wake-up crosses cores
//! into an idle CPU.  It does *not* help buffered streaming, where the host
//! controller keeps transferring while the CPU sleeps.
//!
//! \c /dev/cpu_dma_latency is root-writable by default; \c kame/70-kame.rules
//! opens it to the instrument group.  A request that cannot be established
//! says so once on stderr and stays inactive rather than pretending.
//! Non-Linux platforms compile to a no-op that reports isActive() == false.
class DECLSPEC_KAME XCPULatencyRequest {
public:
    //! \arg target_us largest tolerable C-state exit latency, in microseconds.
    explicit XCPULatencyRequest(int32_t target_us) noexcept;
    ~XCPULatencyRequest();
    XCPULatencyRequest(const XCPULatencyRequest&) = delete;
    XCPULatencyRequest &operator=(const XCPULatencyRequest&) = delete;
    //! \return true if the kernel is actually honouring this request.
    bool isActive() const noexcept {return m_fd >= 0;}
private:
    int m_fd = -1;
};

//! Latency target, in microseconds, requested on the command line
//! (\c --cpulatency).  Negative means "leave the governor alone", the default.
DECLSPEC_KAME extern int32_t g_cpuLatencyTargetUS;

//! round value to the nearest 10s. ex. 42.3 to 10, 120 to 100
DECLSPEC_KAME double roundlog10(double val);
//! round value within demanded precision.
//! ex. 38.32, 40 to 30, 0.4234, 0.01 to 0.42
DECLSPEC_KAME double setprec(double val, double prec) noexcept;

#ifdef _MSC_VER
    #define snprintf(fmt, len, ...) _snprintf_s(fmt, len, len - 1, __VA_ARGS__)
#endif

//! convert control characters to visible (ex. \xx).
DECLSPEC_KAME XString dumpCString(const char *cstr);

//! \sa printf()
DECLSPEC_KAME XString formatString(const char *format, ...)
#if defined __GNUC__ || defined __clang__
    __attribute__ ((format(printf,1,2)));
#endif
;
DECLSPEC_KAME XString formatString_tr(const char *format_i18n_noop, ...)
#if defined __GNUC__ || defined __clang__
    __attribute__ ((format(printf,1,2)));
#endif
;
DECLSPEC_KAME XString formatDouble(const char *fmt, double val);
//! validator
//! throws XKameError
//! \sa XValueNode
DECLSPEC_KAME void formatDoubleValidator(XString &fmt);

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
struct DECLSPEC_KAME X86CPUSpec {
	X86CPUSpec();
	unsigned int verSSE;
	bool hasMonitor;
	unsigned int monitorSizeSmallest;
	unsigned int monitorSizeLargest;
};
DECLSPEC_KAME extern const X86CPUSpec cg_cpuSpec;
#endif

//---------------------------------------------------------------------------
#endif
