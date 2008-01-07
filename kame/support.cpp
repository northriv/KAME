/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include <errno.h>
#include <fcntl.h>
#include <string.h>

bool g_bLogDbgPrint;
bool g_bMLockAlways;
bool g_bUseMLock;

#include <iostream>
#include <fstream>

#include <thread.h>
#define KAME_LOG_FILENAME "/tmp/kame.log"

static std::ofstream g_debugofs(KAME_LOG_FILENAME, std::ios::out);
static XMutex g_debug_mutex;

#include <klocale.h>
#include "support.h"
#include "xtime.h"
#include "measure.h"
#include "threadlocal.h"

#if defined __linux__ || defined MACOSX
#undef TRAP_FPE
#if defined TRAP_FPE && defined __linux__
#include <fpu_control.h>
static void __attribute__ ((constructor)) trapfpe (void)
{
	fpu_control_t cw =
		_FPU_DEFAULT & ~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM);
	_FPU_SETCW(cw);
}
#endif
#endif // __linux__

#ifdef HAVE_LIBGCCPP
static void kame_cleanup(void *obj, void *data) {
	char *msg = (char*)data;
	gErrPrint(QString("Memory Leak! addr=%1, %2")
			  .arg((unsigned int)obj, 0, 16)
			  .arg(msg));
	free(msg);
}
void *kame_gc::operator new(size_t size) {
	char *buf = (char*)malloc(256);
	snprintf(buf, 256, "size=%u, time=%s", 
			 (unsigned int)size, (const char*)XTime::now().getTimeStr().utf8());        
	return ::operator new(size, UseGC, kame_cleanup, buf);
}
void kame_gc::operator delete(void *obj) {
	GC_register_finalizer_ignore_self( GC_base(obj), 0, 0, 0, 0 );
	gc::operator delete(obj);
}
#endif

#ifndef NDEBUG
#ifdef __linux__
#include <execinfo.h>
#endif
void my_assert(const char *file, int line)
{
	XScopedLock<XMutex> lock(g_debug_mutex);
	std::string msg = formatString("assertion failed %s:%d\n",file,line);
	g_debugofs << msg;
	fprintf(stderr, "%s",msg.c_str());
	g_debugofs.flush();
	g_debugofs.close();
#ifdef __linux__
	void *trace[128];
	int n = backtrace(trace, sizeof(trace) / sizeof(trace[0]));
	backtrace_symbols_fd(trace, n, 1);
	int fd = open(KAME_LOG_FILENAME, O_RDWR | O_APPEND);
	backtrace_symbols_fd(trace, n, fd);
	close(fd);
#endif
	abort();
}
#endif // NDEBUG


XKameError::XKameError(const QString &s, const char *file, int line)
	: m_msg(s), m_file(file), m_line(line), m_errno(errno) {
	errno = 0;
}
void
XKameError::print(const QString &header) {
	print(header + m_msg, m_file, m_line, m_errno);
}
void
XKameError::print() {
	print("");
}
void
XKameError::print(const QString &msg, const char *file, int line, int _errno) {
	if(_errno) {
		errno = 0;
		char buf[256];
	#ifdef __linux__
		char *s = strerror_r(_errno, buf, sizeof(buf));
		_gErrPrint(msg + " " + s, file, line);
	#else
		strerror_r(_errno, buf, sizeof(buf));
		if(!errno)
			_gErrPrint(msg + " " + buf, file, line);
		else
			_gErrPrint(msg + " (strerror failed)", file, line);
	#endif
		errno = 0;
	}
	else {
		_gErrPrint(msg, file, line);
	}    
}

const QString &
XKameError::msg() const
{
	return m_msg;
}

double roundlog10(double val)
{
	int i = lrint(log10(val));
	return pow(10.0, (double)i);
}
double setprec(double val, double prec)
{
	double x;

	if(prec <= 1e-100) return val;
	x = roundlog10(prec/2);
	double f = rint(val / x);
	double z = (fabs(f) < (double)0x8fffffff) ? ((int)f) * x : f * x;
	return  z;
}


//---------------------------------------------------------------------------
#include "xtime.h"


void
_dbgPrint(const QString &str, const char *file, int line)
{
	if(!g_bLogDbgPrint) return;
	XScopedLock<XMutex> lock(g_debug_mutex);
	g_debugofs 
		<< (const char*)(QString("0x%1:%2:%3:%4 %5")
						 .arg((unsigned int)threadID(), 0, 16)
						 .arg(XTime::now().getTimeStr())
						 .arg(file)
						 .arg(line)
						 .arg(str))
		.local8Bit()
		<< std::endl;
}
void
_gErrPrint(const QString &str, const char *file, int line)
{
	{
		XScopedLock<XMutex> lock(g_debug_mutex);
		g_debugofs 
			<< (const char*)(QString("Err:0x%1:%2:%3:%4 %5")
							 .arg((unsigned int)threadID(), 0, 16)
							 .arg(XTime::now().getTimeStr())
							 .arg(file)
							 .arg(line)
							 .arg(str))
			.local8Bit()
			<< std::endl;
		fprintf(stderr, "err:%s:%d %s\n", file, line, (const char*)str.local8Bit());
	}
	shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
	if(statusprinter) statusprinter->printError(str);
}
void
_gWarnPrint(const QString &str, const char *file, int line)
{
	{
		XScopedLock<XMutex> lock(g_debug_mutex);
		g_debugofs 
			<< (const char*)(QString("Warn:0x%1:%2:%3:%4 %5")
							 .arg((unsigned int)threadID(), 0, 16)
							 .arg(XTime::now().getTimeStr())
							 .arg(file)
							 .arg(line)
							 .arg(str))
			.local8Bit()
			<< std::endl;
		fprintf(stderr, "warn:%s:%d %s\n", file, line, (const char*)str.local8Bit());
	}
	shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
	if(statusprinter) statusprinter->printWarning(str);
}

#define SNPRINT_BUF_SIZE 128
#include <stdarg.h>
#include <vector>

std::string
formatString(const char *fmt, ...)
{
	va_list ap;
	int buf_size = SNPRINT_BUF_SIZE;
	std::vector<char> buf;
	for(;;) {
		buf.resize(buf_size);
		int ret;

		va_start(ap, fmt);

		ret = vsnprintf(&buf[0], buf_size, fmt, ap);
		va_end(ap);

		if(ret < 0) throw XKameError(KAME::i18n("Mal-format conversion."), __FILE__, __LINE__);
		if(ret < buf_size) break;
  
		buf_size *= 2;
	}
	return std::string((char*)&buf[0]);
}

std::string formatDouble(const char *fmt, double var)
{
	char cbuf[SNPRINT_BUF_SIZE];
	if(strlen(fmt) == 0) {
		snprintf(cbuf, sizeof(cbuf), "%.12g", var);
		return std::string(cbuf);
	}
  
	if(!strncmp(fmt, "TIME:", 5)) {
		XTime time;
		time += var;
		if(fmt[5]) 
			return time.getTimeFmtStr(fmt + 5, false);
		else
			return time.getTimeStr(false);
	}
	snprintf(cbuf, sizeof(cbuf), fmt, var);
	return std::string(cbuf);
}
void formatDoubleValidator(std::string &fmt) {
	if(fmt.empty()) return;

	std::string buf(QString(fmt).latin1());

	if(!strncmp(buf.c_str(), "TIME:", 5)) return;

	int arg_cnt = 0;
	for(unsigned int pos = 0;;) {
		pos = buf.find('%', pos);
		if(pos == std::string::npos) break;
		pos++;
		if(buf[pos] == '%') {
			continue;
		}
		arg_cnt++;
		if(arg_cnt > 1) {
			throw XKameError(KAME::i18n("Illegal Format, too many %s."), __FILE__, __LINE__);
		}
		char conv;
		if((sscanf(buf.c_str() + pos, "%*[+-'0# ]%*f%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%*[+-'0# ]%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%*f%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%c", &conv) != 1)) {
			throw XKameError(KAME::i18n("Illegal Format."), __FILE__, __LINE__);                
		}
		if(std::string("eEgGf").find(conv) == std::string::npos)
			throw XKameError(KAME::i18n("Illegal Format, no float conversion."), __FILE__, __LINE__);  
	}
	if(arg_cnt == 0)
		throw XKameError(KAME::i18n("Illegal Format, no %."), __FILE__, __LINE__);
}

std::string dumpCString(const char *cstr)
{
	std::string buf;
	for(; *cstr; cstr++) {
		if(isprint(*cstr))
			buf.append(1, *cstr);
		else {
			char s[5];
			snprintf(s, 5, "\\x%02x", (unsigned int)(int)*cstr);
			buf.append(s);
		}
	}
	return buf;
}

#include <qdeepcopy.h>

namespace KAME {
	static XThreadLocal<unsigned int> stl_random_seed;
	static XMutex i18n_mutex;
	unsigned int rand() {
		return rand_r(&(*stl_random_seed));
	}
//! thread-safe version of i18n().
//! this is not needed in QT4 or later.
	QString i18n(const char* eng)
	{
		XScopedLock<XMutex> lock(i18n_mutex);
		return QDeepCopy<QString>(::i18n(eng));
	}
}

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
X86CPUSpec::X86CPUSpec() {
	uint32_t stepinfo, features_ext, features;
	asm volatile("push %%ebx; cpuid; pop %%ebx"
	: "=a" (stepinfo), "=c" (features_ext), "=d" (features) : "a" (0x1));
	verSSE = (features & (1uL << 25)) ? 1 : 0;
	if(verSSE && (features & (1uL << 26)))
		verSSE = 2;
	if((verSSE == 2) && (features_ext & (1uL << 0)))
		verSSE = 3;
#ifdef MACOSX
	hasMonitor = false;
#else 
	hasMonitor = (verSSE == 3) && (features_ext & (1uL << 3));
#endif
	monitorSizeSmallest = 0L;
	monitorSizeLargest = 0L;
	if(hasMonitor) {
		uint32_t monsize_s, monsize_l;
		asm volatile("push %%ebx; cpuid; mov %%ebx, %%ecx; pop %%ebx"
		: "=a" (monsize_s), "=c" (monsize_l) : "a" (0x5) : "%edx");
		monitorSizeSmallest = monsize_s;
		monitorSizeLargest = monsize_l;
	}
	fprintf(stderr, "SSE%u, monitor=%u, mon_smallest=%u, mon_larget=%u\n"
		, verSSE, (unsigned int)hasMonitor, monitorSizeSmallest, monitorSizeLargest);
}
const X86CPUSpec cg_cpuSpec;
#endif

	
	
	
