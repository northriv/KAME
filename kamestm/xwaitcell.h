/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
// =====================================================================
// xwaitcell.h — timed wait-on-address primitive (futex-style).
//
// XWaitCell is a sibling of XCondition (xthread.h, which re-exports
// this header), but MUTEX-LESS on macOS.  Where XCondition wraps a
// {mutex + condition variable} pair (pthread_cond → __psynch_cvwait on
// macOS), XWaitCell wraps a single 32-bit word that the kernel
// compares-and-waits on (__ulock_wait, the same primitive
// os_unfair_lock / libdispatch are built on — far lighter than the
// psynch condvar machinery).  The kernel value-compare is what closes
// the lost-wakeup window that XCondition needs its mutex for, so no
// mutex is carried on the macOS path.
//
// This is deliberately a tiny, Qt-free, dependency-light, header-only
// file: it is embedded by value inside Node::NegotiationCounter's
// per-thread sleep slots (transaction.h, via transaction_detail.h),
// which must not drag the full xthread.h / Qt surface into every STM
// translation unit.
//
// Usage protocol (generation counter — see negotiate_sleep):
//   uint32_t g = cell.gen();        // read BEFORE publishing any
//                                   // side-band state
//   ...publish op_kind/stamp...     // a wake() racing in here bumps
//                                   // the generation, so wait() below
//                                   // returns at once — no lost wake
//   cell.wait(g, usec);             // sleeps only while gen == g
// Waker:
//   cell.wake_one();                // advance generation + wake one
//
// Backend selection: KAME_XWAITCELL_ULOCK (default 1 on Apple, 0
// elsewhere).  Override -DKAME_XWAITCELL_ULOCK=0 to force the portable
// std mutex + condvar fallback on macOS too (A/B the free lunch).
// =====================================================================
#ifndef XWAITCELL_H
#define XWAITCELL_H

#include <atomic>
#include <cstdint>

#ifndef KAME_XWAITCELL_ULOCK
#  if defined(__APPLE__)
#    define KAME_XWAITCELL_ULOCK 1
#  else
#    define KAME_XWAITCELL_ULOCK 0
#  endif
#endif

#if KAME_XWAITCELL_ULOCK
#  include <cerrno>   // ETIMEDOUT
// __ulock_wait / __ulock_wake are stable libSystem SPI present since
// macOS 10.12 (the substrate of os_unfair_lock / libdispatch), exported
// from the always-linked libSystem — no extra link flag, works on both
// Apple Silicon and Intel.  Not declared in any public SDK header, so
// declare them here.  UL_COMPARE_AND_WAIT (op 1) is the 32-bit
// FUTEX_WAIT analogue: the kernel reads the word at @addr, compares it
// to @value, and blocks only if equal.  ULF_NO_ERRNO makes the call
// return the negated error code (e.g. -ETIMEDOUT) instead of touching
// errno.  __ulock_wait's timeout is in MICROSECONDS (0 = wait forever).
extern "C" {
    int __ulock_wait(uint32_t operation, void *addr, uint64_t value,
                     uint32_t timeout_us);
    int __ulock_wake(uint32_t operation, void *addr, uint64_t wake_value);
}
#else
#  include <mutex>
#  include <condition_variable>
#  include <chrono>
#endif

//! Timed wait-on-address primitive; sibling of XCondition (xthread.h).
//! Mutex-less on macOS (\sa xwaitcell.h header comment for the
//! generation-counter usage protocol).
class XWaitCell {
public:
    XWaitCell() noexcept = default;
    XWaitCell(const XWaitCell &) = delete;
    XWaitCell &operator=(const XWaitCell &) = delete;

    //! Current generation.  Read this BEFORE publishing any side-band
    //! state, then pass the value to wait(): a wake() that races in
    //! after this read advances the generation, so the value-compare in
    //! wait() returns immediately rather than sleeping on a stale
    //! generation — the lost-wakeup window is closed without a mutex.
    uint32_t gen() const noexcept {
        return m_word.load(std::memory_order_acquire);
    }

    //! Block while the generation still equals \a g, for at most \a
    //! usec microseconds (0 = return immediately, NOT forever).
    //! \return true if woken / the generation changed; false on
    //! timeout.  Spurious wakeups are possible — the caller must
    //! re-check its own condition.
    bool wait(uint32_t g, unsigned usec) noexcept;

    //! Advance the generation (so a waiter that has read g() but not
    //! yet entered wait() bails out of sleeping) and wake at most one
    //! sleeper.
    void wake_one() noexcept;

private:
    std::atomic<uint32_t> m_word{0};
#if !KAME_XWAITCELL_ULOCK
    std::mutex m_mtx;
    std::condition_variable m_cv;
#endif
};

#if KAME_XWAITCELL_ULOCK

inline bool XWaitCell::wait(uint32_t g, unsigned usec) noexcept {
    if(usec == 0)
        return m_word.load(std::memory_order_acquire) != g;
    // UL_COMPARE_AND_WAIT (=1) | ULF_NO_ERRNO (=0x01000000).
    constexpr uint32_t OP = 1u | 0x01000000u;
    int r = __ulock_wait(OP, static_cast<void *>(&m_word),
                         static_cast<uint64_t>(g), usec);
    // With ULF_NO_ERRNO: r >= 0 → woken or value-mismatch; r < 0 → the
    // negated errno.  Only a timeout maps to "false"; any other wake
    // reason (incl. spurious / EINTR) is reported as woken so the
    // caller re-checks its condition.
    return r != -ETIMEDOUT;
}

inline void XWaitCell::wake_one() noexcept {
    // Bump the generation FIRST so a waiter still between gen() and
    // wait() observes the change and skips sleeping; then wake any
    // waiter already parked on the old value.
    m_word.fetch_add(1, std::memory_order_release);
    constexpr uint32_t OP = 1u | 0x01000000u;
    __ulock_wake(OP, static_cast<void *>(&m_word), 0);
}

#else // portable std mutex + condvar fallback

inline bool XWaitCell::wait(uint32_t g, unsigned usec) noexcept {
    std::unique_lock<std::mutex> lk(m_mtx);
    if(m_word.load(std::memory_order_relaxed) != g)
        return true;
    if(usec == 0)
        return false;
    return m_cv.wait_for(lk, std::chrono::microseconds(usec),
                         [&]{ return m_word.load(std::memory_order_relaxed) != g; });
}

inline void XWaitCell::wake_one() noexcept {
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_word.fetch_add(1, std::memory_order_relaxed);
    }
    m_cv.notify_one();
}

#endif

#endif // XWAITCELL_H
