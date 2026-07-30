/***************************************************************************
    Standalone test support header — replaces support.h + xtime.h
    for building transaction tests without Qt/KDE dependencies.
    Uses only C++17 standard library.
 ***************************************************************************/
#ifndef supportH
#define supportH

// --- Minimal support.h replacement ---

#ifndef DECLSPEC_KAME
    #define DECLSPEC_KAME
#endif
#ifndef DECLSPEC_MODULE
    #define DECLSPEC_MODULE
#endif
#ifndef DECLSPEC_SHARED
    #define DECLSPEC_SHARED
#endif

// Platform threading: pthread on POSIX, fall back to C++11 thread_local
// on Windows (no QThread dependency in the standalone tests).
#if !(defined __WIN32__ || defined WINDOWS || defined _WIN32)
    #include <pthread.h>
    #include <sys/mman.h>
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
#include <cmath>
#include <cstdio>

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
struct XKameError : public std::runtime_error {
    virtual ~XKameError() throw() {}
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

#endif // supportH

// --- xtime.h replacement using std::chrono ---

#ifndef XTIME_H_
#define XTIME_H_

#include <chrono>
#include <thread>
#include <cmath>
#include <cstdint>

using namespace std::chrono;

#ifndef NDEBUG
//! Mirrors the debug-only sleep-in-transaction detector in the real xtime.h --
//! see its doc block there.  Needed here and not merely stubbed out, because
//! this harness is the only place the mechanism can actually be exercised: the
//! Qt build is not available in every environment, and transaction_impl.h
//! installs the reporter unconditionally in debug builds.
extern void (*g_sleep_in_transaction_reporter)(
    unsigned int ms, const void *caller) noexcept;
void enterSleepInTransactionOK() noexcept;
void leaveSleepInTransactionOK() noexcept;
struct ScopedSleepInTransactionOK {
    ScopedSleepInTransactionOK() noexcept {enterSleepInTransactionOK();}
    ~ScopedSleepInTransactionOK() noexcept {leaveSleepInTransactionOK();}
};
int _sleep_in_tx_ok_depth() noexcept;
#endif

//! Deliberately NOT inline, matching the real xtime.cpp: the detector above
//! deduplicates on `__builtin_return_address(0)`, which names the caller only
//! while this stays a real out-of-line frame.  Inlined at -O3 it named a
//! libsystem frame instead and every call site collapsed onto one key.
void msecsleep(unsigned int ms) noexcept;

using timestamp_t = uint64_t;

inline timestamp_t timeStamp() noexcept {
    return duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count();
}

inline timestamp_t timeStampCountsPerMilliSec() noexcept {
    return 1000; // microseconds per ms
}

class XTime {
public:
    XTime() noexcept : tv_sec(0), tv_usec(0) {}
    XTime(long sec, long usec) noexcept : tv_sec(sec), tv_usec(usec) {}

    double operator-(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) + (tv_usec - x.tv_usec) * 1e-6;
    }
    long diff_usec(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) * 1000000L + (tv_usec - x.tv_usec);
    }
    long diff_msec(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) * 1000L + ((tv_usec - x.tv_usec) / 1000L);
    }
    long diff_sec(const XTime &x) const noexcept {
        return tv_sec - x.tv_sec;
    }
    XTime &operator+=(double sec_d) noexcept {
        long sec = std::floor(sec_d + tv_sec + 1e-6 * tv_usec);
        long usec = std::lrint(1e6 * (tv_sec - sec + sec_d) + tv_usec);
        tv_sec = sec;
        tv_usec = usec;
        return *this;
    }
    bool operator<(const XTime &x) const noexcept {
        return (tv_sec < x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec < x.tv_usec));
    }
    bool operator!() const noexcept {
        return (tv_sec == 0) && (tv_usec == 0);
    }
    bool isSet() const noexcept {
        return (tv_sec != 0) || (tv_usec != 0);
    }
    long sec() const noexcept { return tv_sec; }
    long usec() const noexcept { return tv_usec; }

    static XTime now() noexcept {
        auto tp = system_clock::now();
        auto dur = tp.time_since_epoch();
        auto secs = duration_cast<seconds>(dur);
        auto usecs = duration_cast<microseconds>(dur) - duration_cast<microseconds>(secs);
        return XTime(secs.count(), usecs.count());
    }

    XString getTimeStr(bool = true) const { return {}; }
    XString getTimeFmtStr(const char *, bool = true) const { return {}; }

private:
    long tv_sec;
    long tv_usec;
};

#endif // XTIME_H_
