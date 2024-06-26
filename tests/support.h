/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
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

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #define USE_QTHREAD
    #include <QThread>
    #include <thread>
#else
    #include <pthread.h>
    #define USE_PTHREAD
#endif

#include <cassert>
#ifdef NDEBUG
#define DEBUG_XTHREAD 0
#else
#define DEBUG_XTHREAD 1
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
#include <algorithm>
#include <math.h>
#include <stdio.h>

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

#include <string>
typedef std::string XString;
#include <stdexcept>
//! Base of exception
struct XKameError : public std::runtime_error {
	virtual ~XKameError() throw() {}
	//! errno is read and cleared after a construction
	XKameError(const XString &s, const char *file, int line);
	void print();
	void print(const XString &header);
	static void print(const XString &msg, const char *file, int line, int errno_);
	const XString &msg() const;
	virtual const char* what() const throw();
private:
	const XString m_msg;
	const char *const m_file;
	const int m_line;
	const int m_errno;
};

//---------------------------------------------------------------------------
#endif
