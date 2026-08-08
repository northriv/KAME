/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#ifndef _MSC_VER
    #include <unistd.h>
#endif
#ifdef __linux__
    #include <sys/resource.h>
    #include <stdlib.h>
    #include <atomic>
#endif

#include "support.h"

bool g_bLogDbgPrint;
bool g_bUseOverpaint;
bool g_bUseMLock;

bool isMemLockAvailable() noexcept {
    if( !g_bUseMLock) return false;
#if defined __linux__
    // --nomlock was the ONLY thing this used to consult, so every caller
    // believed page pinning had happened when in fact all five mlock() call
    // sites discard their return value.  On Linux an unprivileged process is
    // capped by RLIMIT_MEMLOCK — commonly 8 MiB on systemd distros, 64 KiB on
    // older ones and in many containers — and anything past that fails with
    // ENOMEM, so the realtime DSO record buffers were quietly pageable while
    // this function still reported success.  Probe for real, once, and report
    // the limit so the user can act on it.  (The probe cannot tell whether a
    // *particular* later pin will fit; announcing the ceiling is the honest
    // thing this function can do.)
    static std::atomic<int> s_probed{-1};
    int p = s_probed.load(std::memory_order_relaxed);
    if(p < 0) {
        long pg = sysconf(_SC_PAGESIZE);
        if(pg <= 0) pg = 4096;
        void *probe = nullptr;
        p = 0;
        if(posix_memalign( &probe, (std::size_t)pg, (std::size_t)pg) == 0) {
            if(::mlock(probe, (std::size_t)pg) == 0) {
                ::munlock(probe, (std::size_t)pg);
                p = 1;
            }
            free(probe);
        }
        struct rlimit rl = {};
        if( !p) {
            fprintf(stderr, "kame: mlock() is unavailable (%s); realtime buffers will not be"
                " pinned.  Pass --nomlock to silence this.\n", strerror(errno));
        }
        else if((getrlimit(RLIMIT_MEMLOCK, &rl) == 0) && (rl.rlim_cur != RLIM_INFINITY)
                && (rl.rlim_cur < 64u * 1024 * 1024)) {
            fprintf(stderr, "kame: RLIMIT_MEMLOCK is only %lu bytes — pins larger than that"
                " will fail silently.  Raise it (ulimit -l, /etc/security/limits.d, or"
                " LimitMEMLOCK= in a systemd unit).\n", (unsigned long)rl.rlim_cur);
        }
        s_probed.store(p, std::memory_order_relaxed);
    }
    return p != 0;
#else
    return true;
#endif
}

int32_t g_cpuLatencyTargetUS = -1;

XCPULatencyRequest::XCPULatencyRequest(int32_t target_us) noexcept {
#if defined __linux__
    if(target_us < 0) return; //!< Explicitly disabled; leave the governor alone.
    // The constraint lives for exactly as long as this descriptor is open, so
    // the fd -- not the write -- is the request.  O_CLOEXEC matters: KAME
    // spawns Jupyter consoles and terminal emulators, and an inherited copy
    // would keep the CPUs awake long after KAME exited.
    int fd = ::open("/dev/cpu_dma_latency", O_WRONLY | O_CLOEXEC);
    if(fd < 0) {
        fprintf(stderr, "kame: cannot open /dev/cpu_dma_latency (%s); CPU idle states are"
            " left as-is.  It is root-writable by default — install"
            " kame/70-kame.rules and join the instrument group to use --cpulatency.\n",
            strerror(errno));
        return;
    }
    if(::write(fd, &target_us, sizeof(target_us)) != (ssize_t)sizeof(target_us)) {
        fprintf(stderr, "kame: cannot write the PM-QoS target (%s); CPU idle states are"
            " left as-is.\n", strerror(errno));
        ::close(fd);
        return;
    }
    m_fd = fd;
    fprintf(stderr, "kame: CPU idle exit-latency capped at %d us for this process.\n",
        (int)target_us);
#else
    (void)target_us; //!< No equivalent knob on macOS/Windows.
#endif
}
XCPULatencyRequest::~XCPULatencyRequest() {
#if defined __linux__
    if(m_fd >= 0)
        ::close(m_fd); //!< Releases the constraint.
#endif
}

#include <iostream>
#include <fstream>

#include "xthread.h"

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
	#define KAME_LOG_FILENAME "kame.log"
	static std::ofstream g_debugofs(KAME_LOG_FILENAME, std::ios::out);
#else
	#include <unistd.h>
	#include <string>
	// Was a fixed "/tmp/kame.log".  In a world-writable sticky directory that
	// is one shared name for every user on the machine: the second user
	// cannot truncate the first user's file, the ofstream silently enters
	// failbit, and every --logging write after that is discarded with no
	// diagnostic.  (It is also a classic symlink-in-/tmp target.)  Qualify by
	// uid — not by pid, so the path stays predictable for someone who just
	// wants to tail it — and honour $TMPDIR.  This runs during static
	// initialization, so it must not touch Qt.
	static std::string kameLogFilename() {
		const char *tmp = getenv("TMPDIR");
		if( !tmp || !tmp[0]) tmp = "/tmp";
		return std::string(tmp) + "/kame-" + std::to_string((unsigned long)getuid()) + ".log";
	}
	static std::ofstream g_debugofs(kameLogFilename().c_str(), std::ios::out);
#endif
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
        // Was sync(2) here.  That flushes EVERY mounted filesystem, not this
        // log — on Linux it blocks the calling thread (holding g_debug_mutex,
        // which every dbgPrint/gWarnPrint also wants) until the whole page
        // cache is written back, so one error during a large write stalls the
        // acquisition threads for as long as the disk takes.  `std::endl`
        // above already flushed the stream into the kernel, which is what
        // actually protects the log against a KAME crash; surviving a power
        // cut is not worth a system-wide writeback per error message.
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
		// Parse the printf conversion spec explicitly:
		//     %[flags][width][.precision][length]<conv>
		// This used to be a chain of four sscanf() attempts, which depended on
		// how far the C library pushes back a partially-matched %f.  glibc and
		// Darwin libc differ there, and on glibc every precision-bearing
		// exponent format (%.3e, %.5e, %12.4e, %.3E) fell through all four
		// attempts and was rejected as "no float conversion" — so a graph tic
		// label format that works on macOS was refused on Linux, including
		// when it arrived from an existing .kam file.
		const char *p = buf.c_str() + pos;
		p += strspn(p, "+-'0# ");                       //!< flags
		p += strspn(p, "0123456789");                   //!< width
		if( *p == '.') {
			p++;
			p += strspn(p, "0123456789");               //!< precision
		}
		p += strspn(p, "lLhqjzt");                      //!< length modifier
		char conv = *p;
		if( !conv)
            throw XKameError(i18n_noncontext("Illegal Format."), __FILE__, __LINE__);
		if(std::string("eEgGfFaA").find(conv) == std::string::npos)
            throw XKameError(i18n_noncontext("Illegal Format, no float conversion."), __FILE__, __LINE__);
		pos = p - buf.c_str();
	}
	if(arg_cnt == 0)
        throw XKameError(i18n_noncontext("Illegal Format, no %."), __FILE__, __LINE__);
}

XString dumpCString(const char *cstr) {
	XString buf;
	for(; *cstr; cstr++) {
		// `char` is signed on x86 Linux/macOS, so every byte >= 0x80 was a
		// negative int here.  isprint() with a negative argument other than
		// EOF is undefined behaviour, and `(unsigned int)(int)*cstr` produced
		// e.g. 0xffffff80 for 0x80, which "%02x" then printed as "ffffff80"
		// — the interface debug log showed the wrong value for every
		// non-ASCII byte of binary instrument traffic.  Widen through
		// `unsigned char` first.
		unsigned char c = static_cast<unsigned char>( *cstr);
		if(isprint(c))
			buf.append(1, *cstr);
		else {
            char s[5] = {};
			snprintf(s, sizeof(s), "\\x%02x", (unsigned int)c);
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

#include <cpuid.h>

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
        //#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
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
    #if defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
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




