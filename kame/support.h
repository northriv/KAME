/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//Every KAME source must include this header
//---------------------------------------------------------------------------

#ifndef supportH
#define supportH

#define quotedefined(str) #str

#define KAME_DATAFILE_DELIMITER "\t" //Tab for inflexible IGOR

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

#ifdef WORDS_BIGENDIAN
#ifndef __BIG_ENDIAN__
#define __BIG_ENDIAN__
#endif
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#if !defined M_PI
    #define M_PI 3.1415926535897932385
#endif

#include <cassert>
#ifdef NDEBUG
#define DEBUG_XTHREAD 0
#else
#define DEBUG_XTHREAD 1
#endif

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #define USE_QTHREAD
    #define USE_STD_THREAD
    #include <QThread>
    #include <thread>
    DECLSPEC_KAME int mlock(const void *addr, size_t len);
#else
    #include <pthread.h>
    #define USE_PTHREAD
#endif

#include <memory>
using std::unique_ptr;
using std::shared_ptr;
using std::weak_ptr;
using std::enable_shared_from_this;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;
using std::ref;
using std::reference_wrapper;

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

class XString : public std::string {
typedef std::string base_type;
public:
	XString() : base_type() {}
	XString(const char *str) : base_type(str) {}
    XString(const QString &str) : base_type(str.toUtf8().data()) {}
    XString(const base_type &str) : base_type(str) {}
    operator QString() const {return QString::fromUtf8(c_str());}
    XString operator+(const char *s) {return *this + base_type(s);}
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
	virtual ~XKameError() throw() {}

	//! errno is read and cleared after a construction
	XKameError(const XString &s, const char *file, int line);
	void print();
	void print(const XString &header);
	static void print(const XString &msg, const char *file, int line, int errno_);
	const XString &msg() const;
	virtual const char* what() const throw();
private:
	XString m_msg;
	const char * m_file;
	int m_line;
	int m_errno;
};

//! If true, Log all dbgPrint().
extern bool g_bLogDbgPrint;
//! If true, use overpaint feature over OpenGL context.
extern bool g_bUseOverpaint;
//! If true, use mlockall MCL_FUTURE.
extern bool g_bMLockAlways;
//! If true, use mlock.
extern bool g_bUseMLock;

DECLSPEC_KAME bool isMemLockAvailable();

//! round value to the nearest 10s. ex. 42.3 to 10, 120 to 100
DECLSPEC_KAME double roundlog10(double val);
//! round value within demanded precision.
//! ex. 38.32, 40 to 30, 0.4234, 0.01 to 0.42
DECLSPEC_KAME double setprec(double val, double prec);

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
XString formatString_tr(const char *format_i18n_noop, ...)
#if defined __GNUC__ || defined __clang__
    __attribute__ ((format(printf,1,2)));
#endif
;
DECLSPEC_KAME XString formatDouble(const char *fmt, double val);
//! validator
//! throw XKameError
//! \sa XValueNode
DECLSPEC_KAME void formatDoubleValidator(XString &fmt);

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
struct X86CPUSpec {
	X86CPUSpec();
	unsigned int verSSE;
	bool hasMonitor;
	unsigned int monitorSizeSmallest;
	unsigned int monitorSizeLargest;
};
extern const X86CPUSpec cg_cpuSpec;
#endif

//---------------------------------------------------------------------------
#endif
