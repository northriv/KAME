/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "allocator.h"

#ifdef WORDS_BIGENDIAN
#ifndef __BIG_ENDIAN__
#define __BIG_ENDIAN__
#endif
#endif

#ifdef HAVE_LIBGCCPP
//Boehm GC stuff
#define GC_OPERATOR_NEW_ARRAY
#define GC_NAME_CONFLICT
//default size results in falure; "too many root sets", see private/gc_priv.h
#define LARGE_CONFIG
#ifdef __linux__
#define GC_LINUX_THREADS
#define _REENTRANT
#endif
#define GC_DEBUG
#include <gc_cpp.h>
#include <gc_allocator.h>
#if defined __APPLE__
// for buggy pthread library of GC
#define BUGGY_PTHRAD_COND_WAIT_USEC 10000
#define BUGGY_PTHRAD_COND
#endif
class kame_gc : public gc {
public:
	void *operator new(size_t size);
	void operator delete(void *obj);
};
#else
#if defined __WIN32__ || defined WINDOWS
#else
#include <pthread.h>
#endif
#endif

#ifdef NDEBUG
#define ASSERT(expr)
#define C_ASSERT(expr)
#define DEBUG_XTHREAD 0
#else
#define ASSERT(expr) ((expr) ? _my_assert( __FILE__, __LINE__) : 0)
#define C_ASSERT(expr) _my_cassert(sizeof(char [ ( expr ) ? 0 : -1 ]))
inline void _my_cassert(size_t ) {}
int _my_assert(char const*s, int d);
#define DEBUG_XTHREAD 1
#endif

//boost
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
using boost::scoped_ptr;
using boost::shared_ptr;
using boost::weak_ptr;
using boost::enable_shared_from_this;
using boost::static_pointer_cast;
using boost::dynamic_pointer_cast;
#include <boost/ref.hpp>
using boost::ref;
using boost::reference_wrapper;

#include <math.h>
#include <string>
#include <QString>
#include <klocale.h>

class XString : public std::basic_string<char, std::char_traits<char>, allocator<char> > {
typedef std::basic_string<char, std::char_traits<char>, allocator<char> > base_type;
public:
	XString() : base_type() {}
	XString(const char *str) : base_type(str) {}
	XString(const QString &str) : base_type(str.toUtf8().data()) {}
	XString(const base_type &str) : base_type(str) {}
	XString(const std::string &str) : base_type(str.c_str()) {}
	operator QString() const {return QString::fromUtf8(c_str());}
	XString operator+(const char *s) {return *this + base_type(s);}
};

//! Debug printing.
#define dbgPrint(msg) _dbgPrint(msg, __FILE__, __LINE__)
void
_dbgPrint(const XString &str, const char *file, int line);
//! Global Error Message/Printing.
#define gErrPrint(msg) _gErrPrint(msg, __FILE__, __LINE__)
#define gWarnPrint(msg) _gWarnPrint(msg, __FILE__, __LINE__)
void
_gErrPrint(const XString &str, const char *file, int line);
void
_gWarnPrint(const XString &str, const char *file, int line);

#include <stdexcept>
//! Base of exception
struct XKameError : public std::runtime_error {
	virtual ~XKameError() throw() {}
	//! errno is read and cleared after a construction
	XKameError(const XString &s, const char *file, int line);
	void print();
	void print(const XString &header);
	static void print(const XString &msg, const char *file, int line, int _errno);
	const XString &msg() const;
	virtual const char* what() const throw();
private:
	const XString m_msg;
	const char *const m_file;
	const int m_line;
	const int m_errno;
};

//! If true, Log all dbgPrint().
extern bool g_bLogDbgPrint;
//! If true, use mlockall MCL_FUTURE.
extern bool g_bMLockAlways;
//! If true, use mlock.
extern bool g_bUseMLock;
//! round value to the nearest 10s. ex. 42.3 to 10, 120 to 100
double roundlog10(double val);
//! round value within demanded precision.
//! ex. 38.32, 40 to 30, 0.4234, 0.01 to 0.42
double setprec(double val, double prec);

//! convert control characters to visible (ex. \xx).
XString dumpCString(const char *cstr);

//! \sa printf()
XString formatString(const char *format, ...)
__attribute__ ((format(printf,1,2)));
XString formatString_tr(const char *format_i18n_noop, ...)
__attribute__ ((format(printf,1,2)));

XString formatDouble(const char *fmt, double val);
//! validator
//! throw XKameError
//! \sa XValueNode
void formatDoubleValidator(XString &fmt);

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
