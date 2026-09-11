/*
 * support_standalone.cpp — minimal implementations for test-only builds.
 * Replaces support.cpp + xtime.cpp with no Qt/KDE/gettimeofday dependencies.
 */
#include "support_standalone.h"
#include "atomic.h"

#ifndef NDEBUG
// Debug-only sleep-in-transaction detector; see support_standalone.h.
void (*g_sleep_in_transaction_reporter)(
    unsigned int ms, const void *caller) noexcept = nullptr;
static thread_local int s_sleep_in_tx_ok = 0;
int _sleep_in_tx_ok_depth() noexcept {return s_sleep_in_tx_ok;}
void enterSleepInTransactionOK() noexcept {++s_sleep_in_tx_ok;}
void leaveSleepInTransactionOK() noexcept {--s_sleep_in_tx_ok;}
#endif

// Not inline (see support_standalone.h) so the detector's return-address key is
// the caller's.
void msecsleep(unsigned int ms) noexcept {
#ifndef NDEBUG
    if(g_sleep_in_transaction_reporter && !s_sleep_in_tx_ok)
        g_sleep_in_transaction_reporter(ms, __builtin_return_address(0));
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
// The KAME pool allocator lives in its own TU (tests/allocator.cpp) when
// USE_KAME_ALLOCATOR=ON (CMake) or unconditionally in the qmake build.
// Do not #include allocator.cpp here — that was a fragile ODR hack.

X86CPUSpec::X86CPUSpec() {
    uint32_t stepinfo, features_ext, features;
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
    asm volatile("push %%rbx; cpuid; pop %%rbx"
#else
    asm volatile("push %%ebx; cpuid; pop %%ebx"
#endif
    : "=a" (stepinfo), "=c" (features_ext), "=d" (features) : "a" (0x1));
    verSSE = (features & (1uL << 25)) ? 1 : 0;
    if(verSSE && (features & (1uL << 26)))
        verSSE = 2;
    if((verSSE == 2) && (features_ext & (1uL << 0)))
        verSSE = 3;
    hasMonitor = false;
    monitorSizeSmallest = 0;
    monitorSizeLargest = 0;
}
const X86CPUSpec cg_cpuSpec;
#endif

bool g_bUseMLock = false;

int my_assert(char const*s, int d) {
    fprintf(stderr, "Err:%s:%d\n", s, d);
    abort();
    return -1;
}
