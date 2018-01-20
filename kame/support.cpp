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
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#ifndef _MSC_VER
    #include <unistd.h>
    #include <cpuid.h>
#endif

#include "support.h"

bool g_bLogDbgPrint;
bool g_bUseOverpaint;
bool g_bMLockAlways;
bool g_bUseMLock;

bool isMemLockAvailable() noexcept {return g_bUseMLock;}

#include <iostream>
#include <fstream>

#include "xthread.h"

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
	#define KAME_LOG_FILENAME "kame.log"
#else
	#define KAME_LOG_FILENAME "/tmp/kame.log"
#endif

static std::ofstream g_debugofs(KAME_LOG_FILENAME, std::ios::out);
static XMutex g_debug_mutex;

#include "xtime.h"
#include "measure.h"
#include "threadlocal.h"

#if defined __linux__ || defined __APPLE__
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

XKameError::XKameError() : std::runtime_error(""), m_msg(""), m_file(0), m_line(0), m_errno(0) {

}
XKameError::XKameError(const XString &s, const char *file, int line)
	: std::runtime_error(s.c_str()), m_msg(s), m_file(file), m_line(line), m_errno(errno) {
	errno = 0;
}

void
XKameError::print(const XString &header) {
	print(header + m_msg, m_file, m_line, m_errno);
}
void
XKameError::print() {
	print("");
}
void
XKameError::print(const XString &msg, const char *file, int line, int errno_) {
	if( !file) return;
	if(errno_) {
		errno = 0;
		char buf[256] = {};
	#ifdef __linux__
		char *s = strerror_r(errno_, buf, sizeof(buf));
		gErrPrint_redirected(msg + " " + s, file, line);
    #else
        #if defined __WIN32__ || defined WINDOWS || defined _WIN32
            if(strerror_s(buf, sizeof(buf), errno_))
        #else
            if(strerror_r(errno_, buf, sizeof(buf)))
        #endif
                buf[0] = '\0';
        gErrPrint_redirected(msg + " " + buf, file, line);
	#endif
		errno = 0;
	}
	else {
		gErrPrint_redirected(msg, file, line);
	}
}

const XString &
XKameError::msg() const {
	return m_msg;
}

const char* XKameError::what() const noexcept {
	return m_msg.c_str();
}

double roundlog10(double val) {
	int i = lrint(log10(val));
	return pow(10.0, (double)i);
}
double setprec(double val, double prec) noexcept {
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
dbgPrint_redirected(const XString &str, const char *file, int line, bool force_dump) {
    if( !force_dump && !g_bLogDbgPrint) return;
	XScopedLock<XMutex> lock(g_debug_mutex);
	g_debugofs
        << threadID() << (const char*)(QString(":%1:%2:%3 %4")
						 .arg(XTime::now().getTimeStr())
						 .arg(file)
						 .arg(line)
						 .arg(str)).toUtf8().data()
		<< std::endl;
    if(force_dump) {
        shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
        if(statusprinter) statusprinter->printMessage(str, true, file, line);
    }
}
void
gErrPrint_redirected(const XString &str, const char *file, int line) {
	{
		XScopedLock<XMutex> lock(g_debug_mutex);
        fprintf(stderr, "err:%s:%d %s\n", file, line, (const char*)QString(str).toLocal8Bit().data());
		g_debugofs
            << threadID() << (const char*)(QString(":%1:%2:%3 %4")
                             .arg(XTime::now().getTimeStr())
                             .arg(file)
                             .arg(line)
                             .arg(str)).toUtf8().data()
            << std::endl;
#if !defined __WIN32__ && !defined WINDOWS && !defined _WIN32
        sync(); //ensures disk writing.
#endif
	}
	shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
    if(statusprinter) statusprinter->printError(str, true, file, line);
}
void
gWarnPrint_redirected(const XString &str, const char *file, int line) {
	{
		XScopedLock<XMutex> lock(g_debug_mutex);
        fprintf(stderr, "warn:%s:%d %s\n", file, line, (const char*)QString(str).toLocal8Bit().data());
		g_debugofs
            << threadID() << (const char*)(QString(":%1:%2:%3 %4")
                             .arg(XTime::now().getTimeStr())
                             .arg(file)
                             .arg(line)
                             .arg(str)).toUtf8().data()
            << std::endl;
	}
	shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
    if(statusprinter) statusprinter->printWarning(str, false, file, line);
}

#define SNPRINT_BUF_SIZE 1024
#include <stdarg.h>
#include <vector>

static XString
v_formatString(const char *fmt, va_list ap) {
    std::vector<char> buf(SNPRINT_BUF_SIZE);
    int ret = vsnprintf(&buf[0], SNPRINT_BUF_SIZE, fmt, ap);
    if(ret < 0) throw XKameError(i18n_noncontext("Mal-format conversion."), __FILE__, __LINE__);
	return XString((char*)&buf[0]);
}

XString
formatString_tr(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
    XString str = v_formatString(i18n_noncontext(fmt).toUtf8().data(), ap);
	va_end(ap);
	return str;
}

XString
formatString(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	XString str = v_formatString(fmt, ap);
	va_end(ap);
	return str;
}

XString formatDouble(const char *fmt, double var) {
	char cbuf[SNPRINT_BUF_SIZE];
	if(strlen(fmt) == 0) {
		snprintf(cbuf, sizeof(cbuf), "%.12g", var);
        return {cbuf};
	}

	if(!strncmp(fmt, "TIME:", 5)) {
#if !defined __WIN32__ && !defined WINDOWS && !defined _WIN32
        if(isnan(var))
            return "nan";
#endif
		XTime time;
		time += var;
		if(fmt[5])
			return time.getTimeFmtStr(fmt + 5, false);
		else
			return time.getTimeStr(false);
	}
	snprintf(cbuf, sizeof(cbuf), fmt, var);
    return {cbuf};
}
void formatDoubleValidator(XString &fmt) {
	if(fmt.empty()) return;

	XString buf(fmt);

	if( !strncmp(buf.c_str(), "TIME:", 5)) return;

	int arg_cnt = 0;
	for(int pos = 0;;) {
		pos = buf.find('%', pos);
		if(pos == std::string::npos) break;
		pos++;
		if(buf[pos] == '%') {
			continue;
		}
		arg_cnt++;
		if(arg_cnt > 1) {
            throw XKameError(i18n_noncontext("Illegal Format, too many %s."), __FILE__, __LINE__);
		}
		char conv;
		if((sscanf(buf.c_str() + pos, "%*[+-'0# ]%*f%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%*[+-'0# ]%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%*f%c", &conv) != 1) &&
		   (sscanf(buf.c_str() + pos, "%c", &conv) != 1)) {
            throw XKameError(i18n_noncontext("Illegal Format."), __FILE__, __LINE__);
		}
		if(std::string("eEgGf").find(conv) == std::string::npos)
            throw XKameError(i18n_noncontext("Illegal Format, no float conversion."), __FILE__, __LINE__);
	}
	if(arg_cnt == 0)
        throw XKameError(i18n_noncontext("Illegal Format, no %."), __FILE__, __LINE__);
}

XString dumpCString(const char *cstr) {
	XString buf;
	for(; *cstr; cstr++) {
		if(isprint(*cstr))
			buf.append(1, *cstr);
		else {
            char s[5] = {};
			snprintf(s, 5, "\\x%02x", (unsigned int)(int)*cstr);
			buf.append(s);
		}
	}
	return buf;
}

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #include <windows.h>
    int mlock(const void *addr, size_t len) {
        return (VirtualLock((LPVOID)addr, len) != 0) ? 0 : -1;
    }
#endif

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
X86CPUSpec::X86CPUSpec() {
    uint32_t stepinfo, brand, features_ext, features;
#ifdef _MSC_VER
    uint32_t cpuinfo[4];
    __cpuid(reinterpret_cast<int*>(cpuinfo), 0x1);
    stepinfo = cpuinfo[0];
    brand = cpuinfo[1];
    features_ext = cpuinfo[2];
    features = cpuinfo[3];
#else
    __cpuid(0x1, stepinfo, brand , features_ext, features);
//    #if defined __LP64__ || defined __LLP64__
//        asm volatile("push %%rbx; cpuid; pop %%rbx"
//    #else
//        asm volatile("push %%ebx; cpuid; pop %%ebx"
//    #endif
//        : "=a" (stepinfo), "=c" (features_ext), "=d" (features) : "a" (0x1));
#endif
	verSSE = (features & (1uL << 25)) ? 1 : 0;
	if(verSSE && (features & (1uL << 26)))
		verSSE = 2;
	if((verSSE == 2) && (features_ext & (1uL << 0)))
		verSSE = 3;
#ifdef __APPLE__
	hasMonitor = false;
#else
	hasMonitor = (verSSE == 3) && (features_ext & (1uL << 3));
#endif
	monitorSizeSmallest = 0L;
	monitorSizeLargest = 0L;
	if(hasMonitor) {
		uint32_t monsize_s, monsize_l;
        uint32_t cpuinfo[4];
#ifdef _MSC_VER
        __cpuid(reinterpret_cast<int*>(cpuinfo), 0x5);
        monsize_s = cpuinfo[0];
        monsize_l = cpuinfo[2];
#else
        __cpuid(0x5, monsize_s, cpuinfo[1] , monsize_l, cpuinfo[2]);
        //#if defined __LP64__ || defined __LLP64__
        //		asm volatile("push %%rbx; cpuid; mov %%ebx, %%ecx; pop %%rbx"
        //#else
        //		asm volatile("push %%ebx; cpuid; mov %%ebx, %%ecx; pop %%ebx"
        //#endif
        //		: "=a" (monsize_s), "=c" (monsize_l) : "a" (0x5) : "%edx");
#endif
		monitorSizeSmallest = monsize_s;
		monitorSizeLargest = monsize_l;
	}
	fprintf(stderr, "Target: "
#if defined __LP64__
		"x86-64, LP64"
#else
	#if defined __LLP64__
			"x86-64, LLP64"
	#else
			"x86-32"
	#endif
	#if defined __SSE2__
		", SSE2"
	#endif
#endif
		"; Detected: SSE%u\n", verSSE);
}
const X86CPUSpec cg_cpuSpec;
#endif




