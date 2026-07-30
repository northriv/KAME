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
#ifndef TRANSACTION_DETAIL_H
#define TRANSACTION_DETAIL_H

// Internal detail header for transaction.h.  Holds the STM
// framework's machinery that the public API (Node, Snapshot,
// Transaction in transaction.h) is built on top of, but that
// callers should not depend on directly:
//
//   * forward declarations of the public templates;
//   * KAME_CACHE_LINE — ABI-driven L1 dcache line size for
//     `alignas`/`aligned_storage` on hot per-thread / per-shard
//     counters;
//   * NegSite             — per-call-site adaptive negotiation
//                           state machine (take_gate, lease
//                           counters, per-priority retry stats);
//   * LivelockProbe       — per-thread retry-window telemetry
//                           for the LIVELOCK observer;
//   * TidBitset           — per-Transaction observation bitset
//                           for the contention estimate.
//   * popcount_u64        — portable 64-bit popcount helper.
//
// transaction.h #includes this file and then defines Node /
// Snapshot / Transaction on top.  Downstream consumers should
// keep including transaction.h.

// Angle (not "") so INCLUDEPATH order selects support.h, NOT the local
// directory: a KAME build (kame/ first on the path) gets the full,
// Qt-aware kame/support.h; the standalone libkamestm build (kame/ absent)
// gets the Qt-free kamestm/support.h.  A quoted "support.h" would always
// resolve to the kamestm/ shim first (this file lives in kamestm/) and,
// via the shared `supportH` guard, block kame/support.h from ever loading
// — dropping XString-the-class / QString / XKameError / enable_shared_from_this.
#include <support.h>
#include "threadlocal.h"
#include "atomic_smart_ptr.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "xwaitcell.h"   // XWaitCell — futex-style sleep-slot primitive (Qt-free, header-only)
#include <optional>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include "atomic.h"
#include "xtime.h"
#include "transaction_signal.h"
#include "transaction_definitions.h"

// L1 dcache line size by ABI. Used to prevent false sharing on hot
// per-thread / per-shard counters. ABI-driven, not host-detected:
//   x86_64, ARM Cortex-A: 64
//   Apple Silicon (M1+):  128
//   IBM POWER9/10:        128
//   Fujitsu A64FX (SVE):  256 (sector cache)
// std::hardware_destructive_interference_size is ABI-fragile across
// libc++/libstdc++ versions and emits a warning under GCC, so a fixed
// macro is preferred.
#ifndef KAME_CACHE_LINE
  #if defined(__APPLE__) && defined(__aarch64__)
    #define KAME_CACHE_LINE 128
  #elif defined(__powerpc64__) || defined(__POWERPC__)
    #define KAME_CACHE_LINE 128
  #elif defined(__aarch64__) && (defined(__FUJITSU) || defined(__CLANG_FUJITSU))
    #define KAME_CACHE_LINE 256
  #else
    #define KAME_CACHE_LINE 64
  #endif
#endif

namespace Transactional { namespace detail { struct STxNestTag; }}

namespace Transactional {

template <class XN>
class Snapshot;
template <class XN>
class Transaction;
template <class XN>
class ScopedNegotiateLinkage;
template <class XN>
struct ScopedLookupMemoInvalidate;

//! Per-Tx priority, consumed by the negotiator.
//!
//! **What this actually does in a default build**, which is less than the five
//! levels suggest.  The description below used to promise a graduated ladder of
//! per-priority age thresholds; that ladder only operates when
//! `KAME_PER_LINKAGE_PRIVILEGE=0` (the non-default global-privilege mode).  In
//! the default per-Linkage mode the privilege claim opens with a literal
//! `(void)entry_pr;` and the livelock verdict uses
//! `clamp(sig_C*2, 3, hardware_concurrency())` — contention, not priority — so
//! priority does not decide *when* privilege becomes claimable at all.
//!
//! What it does decide:
//!
//!   HIGHEST        — skips negotiation entirely: `if(entry_pr ==
//!                    Priority::HIGHEST) break;` is the first statement of the
//!                    round loop, so the thread never spins and never sleeps.
//!                    Note this buys no deference — nothing makes peers wait
//!                    for it, since only a Reserved (privilege) stamp does that
//!                    and privilege is granted priority-blind.  It is an
//!                    exemption from politeness, and it stops scaling past one
//!                    such thread.  Deliberately not exposed to Python.
//!   LOWEST,        — the "low set".  `now_us_tagged()` folds a lowprio bit
//!   UI_DEFERRABLE,   into their stamps, which makes a privilege stamp they hold
//!   SCRIPTING        *evictable on timeout* (`stamp_is_expired_lowprio`) while
//!                    NORMAL / HIGHEST stamps are immune, and which excludes
//!                    them from the per-Linkage owner-skip lease
//!                    (`_neg_apply_lease`).  This is what keeps a script or a
//!                    UI redraw from pinning privilege against a measurement
//!                    loop.  LOWEST is additionally excluded from the jittered
//!                    gate / lottery in `_negotiate_internal`.
//!   NORMAL         — the baseline: production measurement and driver activity.
//!
//! Intended callers, unchanged: NORMAL for drivers, UI_DEFERRABLE for
//! interactive UI, LOWEST for bulk/analysis, SCRIPTING for external scripting
//! (MCP / AI agents, Python or Ruby via the IPython kernel, future ZMQ
//! handlers) — the last being a one-way trapdoor once set, so a script cannot
//! elevate itself.
//!
//! (The old text also gave SCRIPTING a "1-second threshold".  The value is
//! `min_privilege_age_us(Priority::SCRIPTING) == 1'000` µs = 1 ms — the doc was
//! out by a factor of 1000, and it described a path that the default build does
//! not take.)
enum class Priority {NORMAL = 0, LOWEST, UI_DEFERRABLE, HIGHEST, SCRIPTING};
DECLSPEC_KAME void setCurrentPriorityMode(Priority pr);
DECLSPEC_KAME Priority getCurrentPriorityMode();

//! Thrown by `iterate_commit*` when a transaction at a priority whose privilege
//! can be revoked (LOWEST / UI_DEFERRABLE / SCRIPTING) has been retrying for
//! longer than `KAME_STM_LOWPRIO_STARVE_MS`.
//!
//! Derives from `std::runtime_error` rather than KAME's XKameError so kamestm
//! stays Qt-free, and so pybind11 translates it to a plain Python exception for
//! the SCRIPTING caller, which is the one that can most reasonably retry.
//!
//! NORMAL and HIGHEST never see this: their privilege never expires, so they are
//! never revoked, and losing a driver record to STM contention is a semantic no
//! driver expects.
class StarvationTimeoutError : public std::runtime_error {
public:
    explicit StarvationTimeoutError(const std::string &what)
        : std::runtime_error(what) {}
};

//! Called when a transaction at a revocable priority passes the starvation
//! bound.  \a retries and \a age_us describe the victim.
//!
//! A hook rather than a hard throw, because the type the host wants is the
//! host's business and getting it wrong is fatal.  KAME catches `XKameError` at
//! its connector boundaries and nowhere catches `std::runtime_error`; there is no
//! `QApplication::notify` override and no try/catch around `app.exec()`, so
//! throwing a NEW type from a GUI thread — and `main.cpp` puts the whole GUI
//! thread at UI_DEFERRABLE — terminates the process.  Adding catch sites for a
//! new type meant seven thread entry points plus six connector chains.  Instead
//! the host installs ONE handler that throws its own error type, and every
//! catch site it already has works unchanged.
//!
//! **No handler installed is the default, and means no throw** — the transaction
//! keeps retrying exactly as before.  That way enabling
//! `KAME_STM_LOWPRIO_STARVE_MS` can never introduce an unhandled exception on
//! its own; the host opts in by installing a handler it knows it can catch.
//!
//! A handler that returns rather than throwing is allowed and useful: count and
//! log, and let the retry continue.
using StarvationHandler = void (*)(unsigned retries, long long age_us);
DECLSPEC_KAME void setStarvationHandler(StarvationHandler h) noexcept;
DECLSPEC_KAME StarvationHandler starvationHandler() noexcept;
//! Ready-made handler throwing `StarvationTimeoutError`, for hosts (and tests)
//! that want kamestm's own type.
DECLSPEC_KAME void throwStarvationTimeout(unsigned retries, long long age_us);

//! RAII priority change, restored on scope exit including by exception.
//!
//! Exists because `setCurrentPriorityMode` is a persistent thread mode, and a
//! hand-rolled save/restore is exactly what gets skipped on an early return or
//! a throw.  Scope it instead.  (The one previous hand-rolled user,
//! `strict_escalate_if_oldest`, was removed — it had been dead since its gate
//! macro stopped being defined anywhere.)
//!
//! The `Priority` enumerators are NOT ordered by urgency (NORMAL is 0 and
//! SCRIPTING is 4), so this deliberately does not try to "never weaken an
//! enclosing scope" the way `kame::rt_section` does for RT levels — there is no
//! meaningful max.  Plain save and restore; nesting is the caller's business.
//!
//! One exception is enforced: a thread already at SCRIPTING is left alone.
//! That mirrors the one-way trapdoor the Python binding applies, so an
//! external-scripting thread cannot be lifted out of SCRIPTING by C++ code it
//! happens to call into.
class ScopedPriority {
public:
    explicit ScopedPriority(Priority pr) noexcept
        : m_saved(getCurrentPriorityMode()),
          m_armed(m_saved != Priority::SCRIPTING) {
        if(m_armed) setCurrentPriorityMode(pr);
    }
    ~ScopedPriority() noexcept { if(m_armed) setCurrentPriorityMode(m_saved); }
    ScopedPriority(const ScopedPriority &) = delete;
    ScopedPriority &operator=(const ScopedPriority &) = delete;
    //! False when the scope was suppressed (thread is at SCRIPTING).
    bool armed() const noexcept { return m_armed; }
private:
    Priority m_saved;
    bool m_armed;
};

//! Demotes a realtime committer for the duration of a scope, and ONLY a
//! realtime one.
//!
//! Used where a commit hands control to other people's code: the listeners of a
//! marked message.  The committer's priority must not leak into them — a
//! secondary driver's analysis, a scalar-entry update or a recorder write is not
//! realtime work, and at HIGHEST it inherits an exemption from politeness it has
//! no claim to.  Worse, those listeners widen scope: the secondary-driver
//! interface snapshots the entire driver list, which puts two realtime
//! acquisition threads on one Linkage and breaks the invariant HIGHEST rests on.
//!
//! **One-directional on purpose.**  A lowprio committer must NOT be raised to
//! NORMAL here: a script or a UI redraw committing something would then dispatch
//! its listeners at a priority it cannot claim itself, which is an escalation
//! path, not a fix.
class ScopedDemoteRealtime {
public:
    ScopedDemoteRealtime() noexcept
        : m_saved(getCurrentPriorityMode()),
          m_armed(m_saved == Priority::HIGHEST) {
        if(m_armed) setCurrentPriorityMode(Priority::NORMAL);
    }
    ~ScopedDemoteRealtime() noexcept {
        if(m_armed) setCurrentPriorityMode(m_saved);
    }
    ScopedDemoteRealtime(const ScopedDemoteRealtime &) = delete;
    ScopedDemoteRealtime &operator=(const ScopedDemoteRealtime &) = delete;
private:
    Priority m_saved;
    bool m_armed;
};

//! \name Foreign-lock guard (debug builds only)
//!
//! Detects the OTHER direction of the lock/transaction hazard: holding a plain
//! mutex *across* a Snapshot or Transaction, rather than doing I/O inside one.
//! This is the direction that actually deadlocked KAME on 2026-07-10 — the GUI
//! thread slept in STM negotiation while holding an on-screen-object mutex, and
//! a driver's transaction blocked on that same mutex from inside
//! `finishWritingRaw`, so neither could finish and the negotiation HANG
//! watchdog aborted the process.
//!
//! Why a counter in kamestm that kame/ increments, rather than a check in
//! kame/: the hazard is only realised when the STM *sleeps*, and only the STM
//! knows when that happens.  Putting the predicate here keeps the dependency
//! one-way — kamestm exposes a counter and never looks at kame/ — while the
//! check fires at exactly the point where holding the lock becomes fatal
//! (`_negotiate_internal`, the slow path), not at every Snapshot construction,
//! most of which never wait and are perfectly safe under a lock.
//!
//! Debug builds only, on both sides, so a release build is untouched.
//! \{
#ifndef NDEBUG
namespace detail { struct SForeignLockTag; }
//! Tell the STM this thread now holds a plain lock that it must not sleep
//! under.  Nests; pair each call with `leaveForeignLock()`.
DECLSPEC_KAME void enterForeignLock() noexcept;
DECLSPEC_KAME void leaveForeignLock() noexcept;
//! Nesting depth of the above; 0 = safe to negotiate.
DECLSPEC_KAME int foreignLockDepth() noexcept;
#endif
//! \}

//! \name Wait budget
//!
//! A soft, per-thread bound on how long transactional negotiation may
//! spend **waiting** for contention to clear.  Deliberately NOT called a
//! deadline: nothing here guarantees that a transaction *completes* by
//! the limit.  What it bounds is the waiting — the negotiator will not
//! sleep past the limit, and on reaching it stops being patient.  The
//! overshoot is one commit attempt (measured p99.9 768 ns for a
//! whole-tree scope) plus whatever scheduling jitter the OS adds; on a
//! non-realtime kernel there is no worst-case execution time to make a
//! stronger promise from.
//!
//! Stored as an **absolute** µs value on `LivelockProbe::now_us()`'s
//! steady clock, because the negotiator is re-entered ~1.56 times per
//! slow commit and a relative budget would restart on each entry and so
//! bound nothing.  Callers do not see the absolute form — they set a
//! duration via `ScopedWaitBudget` and the scope converts it once.
//!
//! 0 means "no budget", which is the default and is exactly the
//! pre-existing behaviour: every predicate is guarded on the value being
//! nonzero, so a thread that never sets one runs the same code as before.
//!
//! **Invariant to preserve:** a wait budget may only make the negotiator
//! stop waiting earlier (and, later, claim privilege so the attempt that
//! follows wins).  It must never throw and never abort a transaction.
//! Because the value is ambient it is inherited by callees, including
//! library code the caller does not own; that is safe only as long as
//! its sole effect is impatience.  Ending a job is the caller's own
//! decision, expressed by returning false from `iterate_commit_while`.
//! \{
#if KAME_STM_WAIT_BUDGET
DECLSPEC_KAME int64_t currentWaitLimit() noexcept;
DECLSPEC_KAME void setCurrentWaitLimit(int64_t abs_us) noexcept;

//! Per-thread negotiation context: the priority and the wait limit share
//! ONE thread-local slot on purpose.
//!
//! `XThreadLocal<T>::operator*` caches its pointer in a function-local
//! `thread_local` that is distinct per instantiation, so a second
//! `XThreadLocal` of a different type is a second TLS-wrapper call — on
//! macOS a `_tlv_get_addr`, the same call that cost 4 % on the pool
//! allocator's hot path.  Adding `XThreadLocal<int64_t>` alongside
//! `XThreadLocal<Priority>` measured -2.36 % at 8 threads, lower in 7 of 7
//! interleaved reps.  Packed together, `_negotiate_internal`'s existing
//! single read serves both and the budget costs one extra load from a
//! cache line it already has.
struct TxContext {
    Priority priority   = Priority::NORMAL;
    int64_t  wait_limit = 0;   //!< absolute µs on LivelockProbe::now_us(); 0 = none
};
DECLSPEC_KAME const TxContext &currentTxContext() noexcept;
#endif
//! \}

namespace detail {
    //! Per-thread nesting depth of Transaction/Snapshot scopes.  The
    //! per-thread runner counter (RunnerCounterEntry below) picks up
    //! +1 only on the 0 → 1 transition so nested transactions do not
    //! inflate the runner count.
    struct STxNestTag;
    DECLSPEC_KAME extern XThreadLocal<int, STxNestTag> s_tx_nest;

    //! Foreign non-HIGHEST Reserved stamps stripped by a HIGHEST tagger
    //! (tag_as_contender Rule 0).  Always-on so a plain build can verify the
    //! mechanism actually fired — a strip is rare by construction (it needs a
    //! Reserved stamp in the way), so one relaxed fetch_add costs nothing.
    DECLSPEC_KAME extern std::atomic<std::uint64_t> g_priv_strips;

#ifndef NDEBUG
    //! Debug-only, for `transaction_sleep_in_tx_test`: counts reports actually
    //! emitted by `Transactional::warnIfInTransaction`.  Exists because the gate
    //! that matters -- `*s_tx_nest == 0` -- is inside that function, so a test
    //! cannot pin it by substituting a hook of its own.
    DECLSPEC_KAME extern std::atomic<int> s_in_tx_reports;
#endif

    //! Per-thread nesting depth of ReleaseOneCount (sleeping) scopes.
    struct SSleepNestTag;
    DECLSPEC_KAME extern XThreadLocal<int, SSleepNestTag> s_sleep_nest;

} // namespace detail

//! True while a `Transaction` is alive on this thread — the exact predicate for
//! "we are inside a transaction", and the one place that knowledge is published.
//!
//! Exact rather than approximate: `detail::s_tx_nest` is held for a Transaction's
//! whole lifetime (its `AcquireOneCount` is a value member) but only during a
//! Snapshot's *construction* (a ctor local).  So ordinary
//! `Snapshot shot(*this); ... interface()->query(...)` driver code does not
//! register as being in a transaction, which is correct: a Snapshot blocks
//! nothing.
DECLSPEC_KAME bool isInTransaction() noexcept;

#ifndef NDEBUG
//! Debug-only diagnostic: report, once per `where`/`site`, that something
//! happened while a transaction was alive.  No-op outside a transaction.
//!
//! This is the single implementation behind every such check — `XInterface::lock`
//! (interface I/O inside a transaction, driver rule 5) and `msecsleep` (sleeping
//! with a transaction open) both land here rather than each carrying its own copy
//! of the gate, the deduplication and the abort switch.  `kame/support.h` wraps it
//! as `gWarnIfInTransaction(what)`, which fills in `where` from `__FILE__:__LINE__`.
//!
//! Reports rather than aborting by default, because this is a diagnostic and not a
//! safeguard: an unknown number of sites may still reach it through indirection
//! the audit cannot see, and a debug build that aborts on the first one is a debug
//! build nobody runs.  Set `KAME_STM_ABORT_IN_TX=1` when actually hunting one.
//!
//! \param what  a full sentence for a driver author, stating what was done and
//!              why it is wrong; the location and the abort hint are appended.
//! \param where source location, or nullptr; deduplicates and is printed.
//! \param site  fallback deduplication key when there is no source location
//!              (`msecsleep` has only its caller's return address).
DECLSPEC_KAME void warnIfInTransaction(const char *what, const char *where,
                                       const void *site = nullptr) noexcept;
#endif

namespace detail {

    //! Payload-creator slot: create<T>() stores the typed creator
    //! here; Node<XN>::Node() reads and clears it.
    struct TlsPayloadCreatorPtrTag;
    DECLSPEC_KAME extern XThreadLocal<void*, TlsPayloadCreatorPtrTag>
        tls_payload_creator_ptr;

    //! Lamport serial: `SerialGenerator::gen()` / `current()` read+RMW
    //! this.  Default ctor seeds the lower 16-bit thread-ID half from
    //! `ProcessCounter::id()`.
    struct TlsSerial {
        int64_t v;
        TlsSerial() noexcept
            : v(static_cast<int64_t>(ProcessCounter::id())) {}
    };
    DECLSPEC_KAME extern XThreadLocal<TlsSerial> tls_serial;

    //! op_kind discriminator values.  0 (NONE) is the default for
    //! non-piggyback-aware code paths (stand-alone snapshot / release
    //! / insert / SingleTransaction).  The other values mark
    //! Tx-driven sub-operations of a multilevel commit:
    //!   BUNDLE / UNBUNDLE: tree-walk sub-CASes inside a multi-nodal
    //!     commit; tagged inside Node<XN>::bundle() / ::unbundle().
    //!     Multi-nodal commits at the outer iterate_commit /
    //!     commit_loop level also stamp BUNDLE — the per-Linkage flip
    //!     accounting (m_recent_ops_state) and the spin-block kind
    //!     filter treat outer-commit and inner-bundle identically, so
    //!     a separate marker added no information.
    //!   Reserved (= 3): formerly `MultiNodalCommit`, an alias of
    //!     BUNDLE in every production code path (only INSTRUMENT
    //!     counters distinguished it).  Now free for re-use as a
    //!     per-Linkage privilege flag (`m_transaction_started_time`
    //!     kind=3 → "this Tx holds privilege on this Linkage").
    enum class StampKind : uint8_t {
        NONE             = 0,
        BUNDLE           = 1,
        UNBUNDLE         = 2,
        Reserved         = 3,
    };

    //! Per-thread "current operation kind".  Pushed via ScopedOpKind
    //! (below) at bundle/unbundle/commit entry; read by Snapshot's
    //! tag_as_contender to stamp the linkage with the appropriate
    //! kind.  Outside any ScopedOpKind scope the value is NONE.
    struct SCurrentOpKindTag;
    DECLSPEC_KAME extern XThreadLocal<StampKind, SCurrentOpKindTag>
        s_current_op_kind;

    //! RAII helper to push a new op_kind onto the per-thread slot.
    //! Nested usage (e.g. bundle inside a commit, or unbundle during
    //! commit's CASInfo loop) is supported via save/restore in
    //! ctor/dtor.
    struct ScopedOpKind {
        StampKind prev;
        explicit ScopedOpKind(StampKind k) noexcept
            : prev(*s_current_op_kind) { *s_current_op_kind = k; }
        ~ScopedOpKind() noexcept { *s_current_op_kind = prev; }
        ScopedOpKind(const ScopedOpKind &) = delete;
        ScopedOpKind &operator=(const ScopedOpKind &) = delete;
    };

    // (NegSite — per-call-site Neg state — lives outside detail:: below.)

    //! Cacheline-padded "I am currently running a Tx" counter, owned
    //! by exactly one thread. Heap-allocated per thread so increments
    //! are TLS-affine (no cacheline ping-pong even across NUMA nodes).
    //! Replaces the previous single
    //! `alignas(64) atomic<unsigned> s_running` whose ping-pong on
    //! every Tx entry/exit was the K=0 disjoint NUMA-scaling ceiling
    //! on high-core-count x86_64 (128 logical cores) (≈8.3 M Tx/s × 2 atomic RMWs/Tx ≈ 16.6 M ops/s ≈
    //! cross-socket cacheline bandwidth).
#if KAME_ENABLE_RUNNER_DIGEST
    //! Bit-packed peer-readable digest of a thread's negotiation
    //! state.  Owner mutates the per-thread cache `tls_runner_digest`
    //! at ScopedNeg boundaries and publishes the raw uint64 via a
    //! single relaxed atomic store on `RunnerCounterEntry::digest`.
    //! Peer threads (CV-sleep judge, low-probability re-evaluation)
    //! issue one relaxed load + decode the union locally — no per-
    //! field synchronisation needed.  Layout (LSB → MSB, total 64):
    //!   tx_start_us  22  low bits of the Tx's m_started_time µs
    //!                    field; peer computes age via
    //!                    `(now_us - tx_start_us) mod 4.2 s`.
    //!                    Published value is the Snapshot's stamp_us
    //!                    cached at scope-end — no per-publish
    //!                    steady_clock call.
    //!   op_kind       2  detail::StampKind
    //!   gate_history  8  recent gate/sleep decisions, LSB=newest
    //!   consec_succs  8  saturating mirror of SiteState::consec_succs
    //!   consec_fails  8  saturating mirror of SiteState::consec_fails
    //!   take_gate_p1  2  SiteState::take_gate + 1 (0=UNDEF, 1=SLEEP, 2=GATE)
    //!   site_line_lo 12  __LINE__ low 12 bits (4096 sites)
    //!   reserved      2
    //! Idle vs stale: peer should consult `v` first — if v==0 the
    //! thread is between Tx, and the digest's last published values
    //! are simply old.
    union RunnerDigest {
        struct Fields {
            uint64_t tx_start_us  : 22;
            uint64_t op_kind      : 2;
            uint64_t gate_history : 8;
            uint64_t consec_succs : 8;
            uint64_t consec_fails : 8;
            uint64_t take_gate_p1 : 2;
            uint64_t site_line_lo : 12;
            uint64_t reserved     : 2;
        } f;
        uint64_t raw;
        RunnerDigest() noexcept : raw(0) {}
    };
    static_assert(sizeof(RunnerDigest) == sizeof(uint64_t),
                  "RunnerDigest must pack into 64 bits");
#endif // KAME_ENABLE_RUNNER_DIGEST

    //! Per-thread "I'm in a Tx" counter (plus, when enabled, the
    //! peer-readable digest).  One instance per thread, heap-
    //! allocated by `runner_counter_register()` and linked into the
    //! global singly-linked list anchored at `s_runner_entries_head`.
    //!
    //! `v` encoding (back to the simple pre-chunked semantics):
    //!   v == 0  : idle (not currently in a transaction)
    //!   v >= 1  : running (one per nesting depth, but
    //!             AcquireOneCount RAII guarantees v ∈ {0, 1} in
    //!             steady state)
    //!
    //! `AcquireOneCount` does `v.fetch_add(1, release)` on the
    //! outermost transaction (0 → 1); `~AcquireOneCount` does the
    //! reciprocal `fetch_sub` (1 → 0).  `ReleaseOneCount` (CV-sleep
    //! yield) mirrors the same with reversed direction.
    //! `num_threads_running_impl` reads with `v.load(acquire)` and
    //! sums across all entries (release/acquire pairing collapses
    //! cross-socket NUMA staleness).
    //!
    //! Lifetime: heap-allocated by the owning thread on first STM
    //! use → first-touch places the allocation on that thread's
    //! local NUMA node → `fetch_add/sub` is **local** even on high-core-count x86_64
    //! dual-socket.  This is the critical NUMA placement property
    //! that the previous chunked-array design lost — on
    //! high-core-count dual-socket NUMA x86_64 chunked arrays mixed slots across
    //! sockets and per-tx fetch_add became cross-socket (~1 µs),
    //! yielding -37% throughput vs the heap-per-thread layout.
    //!
    //! Entries persist until process exit.  Bounded by **max-ever-
    //! concurrent thread count** (not max-ever-total) via slot reuse:
    //! when a thread exits, its TLS dtor sets `claimed=false`, leaving
    //! the entry available for the next first-time thread to CAS-
    //! claim.  This handles long-running processes with worker
    //! spawn/destroy churn without unbounded leak.
    //!
    //! NUMA-aware reuse (Linux only): each entry stores its
    //! `allocated_node` (set once at construction, from `getcpu(2)`).
    //! `runner_counter_register` does a two-pass scan — first pass
    //! prefers entries on the calling thread's local NUMA; second pass
    //! takes any unclaimed entry as fallback.  Long-term steady-state
    //! drives entries to settle on threads of matching NUMA, restoring
    //! local `fetch_add/sub`.  On non-Linux (Win/Mac), single-pass
    //! blind reuse — no penalty on uniform-memory single-socket
    //! systems where the ARM64 production target lives.
    //!
    //! A process-exit teardown sentinel walks and deletes the list
    //! for leak-detector hygiene.
    //!
    //! Replaces the heap-vector+atomic_shared_ptr design (eliminated
    //! the 27.8% hotspot on x86_64 NUMA and the TLS-teardown race) AND the
    //! intermediate chunked-array design (eliminated the cross-socket
    //! NUMA pitfall of contiguous slot blocks).
    struct alignas(KAME_CACHE_LINE) RunnerCounterEntry {
        // Steady-state values are 0/1 (one per nesting depth, but
        // AcquireOneCount RAII keeps the outer ∈ {0,1}); 64-bit
        // accommodates pathologically deep nesting on roomy hosts.
        // Under compact mode (no lock-free 64-bit atomics) shrink to
        // 32-bit — still > 4 G counts, far beyond any realistic nesting.
#if KAME_STM_COMPACT_STATE
        std::atomic<uint32_t> v{0};
#else
        std::atomic<uint64_t> v{0};
#endif
#if KAME_ENABLE_RUNNER_DIGEST
        //! Placed immediately after `v` (both uint64_t, no implicit
        //! alignment padding) so the _pad formula stays correct on
        //! 32-bit and 64-bit alike.  If declared after `next`/`claimed`/
        //! `allocated_node`, the compiler inserts hidden padding to
        //! bring digest to an 8-byte boundary, causing the struct to
        //! overflow one cache line.
        std::atomic<uint64_t> digest{0};   // raw value of RunnerDigest
#endif
        //! Set once during `runner_counter_register` (just before the
        //! CAS-prepend onto `s_runner_entries_head`); immutable
        //! thereafter.  The publishing CAS uses `acq_rel`, so a
        //! reader that synchronises-with the new head via
        //! `s_runner_entries_head.load(acquire)` automatically sees
        //! this `next` value with no further ordering needed.
        //! Declared `std::atomic<>` for ABI/lint cleanliness; loads
        //! in the traversal hot path are `relaxed`.
        std::atomic<RunnerCounterEntry*> next{nullptr};
        //! `true` while an owning thread holds this entry; `false`
        //! when the entry is free for the next first-time-registering
        //! thread to CAS-claim.  Newly-constructed entries have
        //! `claimed=true` (the allocator is the first owner); the
        //! TLS dtor `RunnerEntryReleaseGuard` flips this back to
        //! `false` on thread exit.
        std::atomic<bool> claimed{true};
        //! Immutable after construction: the NUMA node id (from
        //! `getcpu(2)` on Linux, else -1) of the thread that
        //! originally allocated this entry's memory.  Used by the
        //! reuse path's first pass to prefer entries on the
        //! calling thread's local NUMA.  Reader does not touch this.
        int8_t allocated_node;
        RunnerCounterEntry(int8_t node) noexcept
            : allocated_node(node) {}
#if KAME_ENABLE_RUNNER_DIGEST
        char _pad[KAME_CACHE_LINE - sizeof(decltype(v))
                                  - sizeof(std::atomic<uint64_t>)
                                  - sizeof(std::atomic<RunnerCounterEntry*>)
                                  - sizeof(std::atomic<bool>)
                                  - sizeof(int8_t)];
#else
        char _pad[KAME_CACHE_LINE - sizeof(decltype(v))
                                  - sizeof(std::atomic<RunnerCounterEntry*>)
                                  - sizeof(std::atomic<bool>)
                                  - sizeof(int8_t)];
#endif
    };

#if KAME_ENABLE_RUNNER_DIGEST
    //! Per-thread local cache of the digest.  Owner mutates this on
    //! scope/Tx boundaries; publish atomically to
    //! `RunnerCounterEntry::digest` via the publish path in
    //! ScopedNegotiateLinkage.  Non-atomic — never touched by peers.
    DECLSPEC_KAME extern XThreadLocal<RunnerDigest> tls_runner_digest;
#endif // KAME_ENABLE_RUNNER_DIGEST

    //! Cached raw pointer to this thread's heap-allocated entry.
    //! Set once on first registration (`runner_counter_register`) and
    //! reused for the lifetime of the thread.  Hot-path consumers
    //! (AcquireOneCount, ReleaseOneCount, ScopedNeg digest publish)
    //! dereference it directly — one TLS load + one release/relaxed
    //! fetch_add, no shared_ptr / atomic_shared_ptr operations.
    struct TlsRunnerCounterPtrTag;
    DECLSPEC_KAME extern XThreadLocal<RunnerCounterEntry*, TlsRunnerCounterPtrTag>
        tls_runner_counter_ptr;

    //! Head of the per-thread runner-counter linked list.  Each
    //! thread CAS-prepends its heap-allocated `RunnerCounterEntry`
    //! here on first STM use.  Reader (`num_threads_running_impl`)
    //! walks `head → entry.next → ...`; once an entry is in the
    //! list its `next` pointer is immutable.
    DECLSPEC_KAME extern std::atomic<RunnerCounterEntry*> s_runner_entries_head;

    //! Allocate + register this thread's counter on first call;
    //! return the cached raw pointer thereafter. Defined in
    //! transaction_impl.h.
    DECLSPEC_KAME RunnerCounterEntry& runner_counter_register();

    // All TLS variables migrated to XThreadLocal — no `#define`
    // macro scaffolding remaining.  Access via the unified `*x` /
    // `x->field` pattern at call sites.

    //! libkame-side bodies — keep all module-visible code thin so
    //! modules never instantiate their own copy of the runner-counter
    //! state. (The Apple/Linux DSO-duplication failure mode and the
    //! Windows MSVC dllexport caveats both push us toward routing
    //! every access through libkame symbols.)
    DECLSPEC_KAME RunnerCounterEntry& my_runner_counter_impl();
    DECLSPEC_KAME unsigned int num_threads_running_impl(unsigned int ceiling) noexcept;

    inline RunnerCounterEntry& my_runner_counter() {
        return my_runner_counter_impl();
    }

    //! Sum across all registered threads — early-exits the per-entry
    //! iteration once the partial sum reaches `ceiling`.  Called only
    //! from `negotiate_internal` (gate / lottery / wake decisions) —
    //! never on the K=0 disjoint hot path.
    //!
    //! Hot-path callers compare the result against small thresholds
    //! (typically `KAME_STM_MAX_RUNNERS = 2` or `min_r = 1..2`); for
    //! those, passing the threshold as `ceiling` lets the loop bail
    //! after iterating ~ceiling entries instead of all 128 — VTune
    //! on an x86_64 NUMA server showed the un-capped version costing 30% of
    //! CPU time at 128 threads.
    //!
    //! Default `ceiling = ~0u` reproduces the original "exact sum"
    //! semantics, for callers (lease scaling, wake-count calculation)
    //! that actually use the magnitude.
    inline unsigned int num_threads_running(
        unsigned int ceiling = ~0u) noexcept {
        return num_threads_running_impl(ceiling);
    }
} // namespace detail

// Adaptive-gate / spin-path tuning knobs live in transaction_definitions.h
// (KAME_GATE_K_FAIL, KAME_GATE_K_SUCC, KAME_GATE_FAIL_WINDOW_US,
// KAME_NORMAL_LEASE_US, KAME_SPIN_MAX_US, KAME_SPIN_RECENT_FLIP_US).

//! Per-call-site adaptive state + diagnostics for ScopedNegotiateLinkage.
//!
//! Non-template, namespace-style class: all members are static.  Holds
//! the tri-state `take_gate` machinery (Adaptive struct + per-thread
//! map + cached "current site" pointer), the gate-return sink, the
//! INSTRUMENT-side stat counters, and the RAII Scope helper that primes
//! the TLS slots at ScopedNeg ctor.  Keeping these here (rather than in
//! detail::) gives a single Neg-prefixed namespace for everything
//! related to the gate-return state machine; ScopedNegotiateLinkage<XN>
//! references `NegSite::*` directly.
//!
//! TLS access pattern: function-local thread_local inside DECLSPEC_KAME
//! accessor functions — DSO-portable on all platforms (sidesteps MSVC
//! dllexport-thread_local restriction and the Apple clang arm64
//! template-static-thread_local wrapper-emission bug equally; storage
//! is one slot per program because libkame defines each accessor once).
//!
//! Adaptive::take_gate tri-state:
//!   -1 (UNDEFINED)   : initial state, also post-privilege and
//!                      post-lease-expiry reset state.  Hot path
//!                      decides take_gate by my_kind alone
//!                      (non-NONE → gate, NONE → sleep).
//!    0 (FORCE_SLEEP) : forced sleep, regardless of my_kind.
//!                      Time-leased via `normal_until_us`.
//!    1 (FORCE_GATE)  : forced gate-return, regardless of my_kind
//!                      (NONE included).  Time-leased.
//!
//! Transitions (driven by ScopedNeg::_on_cas_fail / _on_cas_success):
//!   UNDEFINED   → FORCE_SLEEP : K_FAIL gate→fails inside
//!                               FAIL_WINDOW_US (gated calls only —
//!                               i.e. my_kind != NONE).
//!   UNDEFINED   → FORCE_GATE  : K_SUCC gate→succs (gated calls only).
//!   FORCE_SLEEP → FORCE_GATE  : K_SUCC consecutive CAS successes
//!                               (any my_kind — site is "quiet
//!                               enough" to risk all-kind gating).
//!   FORCE_GATE  → FORCE_SLEEP : K_FAIL gate→fails (every caller
//!                               gates in FORCE_GATE).
//!   any FORCE   → UNDEFINED   : lease expiry (in negotiate_internal)
//!                               or privilege observation.
//!
//! Opt-in INSTRUMENT (`-DKAME_ADAPT_INSTRUMENT=1`) now only controls
//! whether per-thread stats are auto-merged into the global aggregator
//! at thread exit (via the AutoMergeStats sentinel) and whether
//! `dump()` is wired into testbench teardown.  The per-site counters
//! themselves are **always on** — they live alongside the adaptive
//! state machine in `NegSite::SiteState`, so live introspection (e.g.
//! via the Python MCP `kame.NegSite.dump()`) sees current data without
//! a special build.
class DECLSPEC_KAME NegSite {
public:
    //! State-machine tuning knobs (overridable via -DKAME_GATE_*).
    static constexpr int GATE_K_FAIL         = KAME_GATE_K_FAIL;
    static constexpr int GATE_K_SUCC         = KAME_GATE_K_SUCC;
    static constexpr int GATE_FAIL_WINDOW_US = KAME_GATE_FAIL_WINDOW_US;
    static constexpr int NORMAL_LEASE_US     = KAME_NORMAL_LEASE_US;

    //! Per-call-site state — unified container for the production
    //! adaptive state machine (take_gate, streak counters, lease) and
    //! the cumulative diagnostic counters (entries/commits/per-peer
    //! breakdowns).  All fields are always-on; the previous
    //! INSTRUMENT-only split (`Adaptive` vs `Stat`) was merged so that
    //! ScopedNeg only ever touches one struct.
    struct SiteState {
        // --- Adaptive state machine (production hot path) ---
        int8_t   take_gate = -1;               // -1=UNDEFINED, 0=SLEEP, 1=GATE
        uint16_t consec_fails = 0;             // streak toward FORCE_SLEEP
        uint16_t consec_succs = 0;             // streak toward FORCE_GATE
        uint64_t last_fail_us = 0;             // start of current fail-streak
        uint64_t normal_until_us = 0;          // lease expiry (both FORCE states)
        uint64_t mode_flips_g2n = 0;           // diagnostic: any → FORCE_SLEEP
        uint64_t mode_flips_n2g = 0;           // diagnostic: FORCE → UNDEFINED
        uint64_t mode_flips_promote = 0;       // diagnostic: FORCE_SLEEP → FORCE_GATE
        // --- Cumulative counters (live, always-on) ---
        uint64_t entries = 0;                  // ScopedNeg ctor count
        uint64_t commits = 0;                  // m_committed at dtor
        uint64_t blocked_by_peer[4] = {};      // gate not taken: normal sleep
        uint64_t gate_returns_by_peer[4] = {}; // gate-return fired
        uint64_t gate_then_cas_fail = 0;       // gate-return → CAS failed
    };

#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    //! INSTRUMENT-only thread-exit sentinel.  Its dtor flushes the
    //! per-thread state_map into the global aggregator so dump() at
    //! process end sees stats from threads that have already exited.
    //! Production builds rely on on-demand merge from dump() instead,
    //! so a thread that runs and never exits before dump() is still
    //! covered if the dump-issuing thread invokes mergeStatsToGlobal()
    //! synchronously.
    struct AutoMergeStats {
        ~AutoMergeStats() noexcept;
    };
#endif

    //! RAII helper: push the active call-site's SiteState pointer onto
    //! the TLS slot.  The hot path in negotiate_internal / _on_cas_*
    //! dereferences `current_state()` directly — avoiding a per-call
    //! unordered_map lookup.  Mirrors detail::ScopedOpKind.
    struct Scope {
        SiteState *prev_state;
        explicit Scope(int line) noexcept
            : prev_state(NegSite::current_state())
        {
            NegSite::current_state() = &NegSite::state_map()[line];
        }
        ~Scope() noexcept {
            NegSite::current_state() = prev_state;
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
    };

    //! TLS accessors — inline wrappers around the `tls_*` members below.
    static std::unordered_map<int, SiteState>& state_map() noexcept;
    static SiteState*& current_state() noexcept;
    //! Sink for the most recent negotiate_internal() gate-return
    //! decision (true if take_gate fired).  Captured at ScopedNeg ctor
    //! into m_was_gate_return for per-scope use by _on_cas_fail /
    //! _on_cas_success.
    static bool& last_was_gate_return() noexcept;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    static AutoMergeStats& auto_merge_stats() noexcept;
#endif

    struct StateMapTag;
    struct CurrentStateTag;
    struct LastWasGateReturnTag;
    DECLSPEC_KAME static XThreadLocal<std::unordered_map<int, SiteState>,
                                       StateMapTag> tls_state_map;
    DECLSPEC_KAME static XThreadLocal<SiteState *, CurrentStateTag>
                                       tls_current_state;
    DECLSPEC_KAME static XThreadLocal<bool, LastWasGateReturnTag>
                                       tls_last_was_gate_return;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    struct AutoMergeStatsTag;
    DECLSPEC_KAME static XThreadLocal<AutoMergeStats, AutoMergeStatsTag>
                                       tls_auto_merge_stats;
#endif

    //! Merge this thread's state_map into the global aggregator.
    //! Always-callable now (no INSTRUMENT guard): the global aggregator
    //! exists in production, this thread's per-line entries get folded
    //! in, and dump() prints them.  Cheap enough to call on demand from
    //! any thread (one mutex + a map merge of <100 entries).
    DECLSPEC_KAME static void mergeStatsToGlobal() noexcept;
    //! Dump a human-readable per-site summary to `fp` (stderr by default).
    //! Calls mergeStatsToGlobal() internally to pull in the caller
    //! thread's state.  Always-callable in production.
    DECLSPEC_KAME static void dump(std::FILE *fp = nullptr) noexcept;

    //! Record a Linkage-level kind flip — i.e., a tag-as-contender by a
    //! DIFFERENT thread with a DIFFERENT op_kind than the previous
    //! tagger on the same Linkage.  Aggregated globally as a 4x4
    //! matrix [prev_kind][curr_kind] for `dump()`, plus a log-binned
    //! histogram of the inter-flip interval (`interval_us`, in µs).
    //! INSTRUMENT-only; folds to a no-op in production builds.
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    DECLSPEC_KAME static void record_linkage_flip(uint8_t prev_kind,
                                                  uint8_t curr_kind,
                                                  uint32_t interval_us) noexcept;
#else
    static void record_linkage_flip(uint8_t, uint8_t, uint32_t) noexcept {}
#endif

    //! Outcome tag for spin-for-same-kind events.  Aggregated as
    //! [outcome] counters and printed by dump().  INSTRUMENT-only.
    enum class SpinOutcome : uint8_t {
        SKIPPED_NO_PERIOD = 0,  // period_us == 0 (Linkage never flipped)
        SKIPPED_COLD      = 1,  // period_us > THRESHOLD (cold Linkage)
        SKIPPED_PAST      = 2,  // age >= period (predicted flip already past)
        SKIPPED_SAME_KIND = 3,  // last_kind == my_kind (already aligned)
        WON               = 4,  // spin saw same kind / Linkage release
        TIMEOUT           = 5,  // spin budget expired
        SKIPPED_THRASHING = 6,  // fs_period_us < sig_C (over-thrashing
                                // regime — flips arrive faster than
                                // thread round-robin, so even one-period
                                // spin can't catch a stable window)
        GATE_RETURN_SAMEKIND = 7, // flip detected, spin gate closed
                                // (COLD / THRASHING), but peer's current
                                // kind matches ours — skip CV-sleep and
                                // retry CAS immediately (cheap
                                // piggyback on peer's same-kind work).
    };
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    DECLSPEC_KAME static void record_spin_event(SpinOutcome o,
                                                uint32_t elapsed_us) noexcept;
#else
    static void record_spin_event(SpinOutcome, uint32_t) noexcept {}
#endif

    //! Outcome tag for the gate-return count-band decision.  Captures
    //! WHERE the observed flip count fell relative to the sweet-spot
    //! band [LOW, effective_HIGH].  INSTRUMENT-only.
    enum class BandOutcome : uint8_t {
        BELOW_LOW   = 0,  // count < LOW           (no periodic evidence)
        IN_BAND     = 1,  // LOW <= count <= HIGH  (gate-return fires)
        ABOVE_HIGH  = 2,  // count > effective_HIGH (hyper-thrashing,
                          // skipped to protect anti-phase)
    };
    //! Record a gate-return band decision keyed by (kind, outcome,
    //! tighten level).  kind: 0=any/outer (mk=NONE), 1=B, 2=U, 3=C.
    //! Aggregated as 4 × 3 × (MAX_TIGHTEN+1) counters by dump().
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    DECLSPEC_KAME static void record_band_event(uint8_t kind,
                                                BandOutcome outcome,
                                                uint8_t tighten) noexcept;
#else
    static void record_band_event(uint8_t, BandOutcome, uint8_t) noexcept {}
#endif

    //! Record gate-return outcome: "in time" (CAS succeeded soon
    //! after the gate-return decision).  latency_us is the wall-clock
    //! delta from the gate-return decision to the CAS success.  Bucketed
    //! into a log-binned histogram.  my_kind: kind of the Tx that
    //! issued the gate-return (1=B, 2=U, 3=C, 0=outer).
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    DECLSPEC_KAME static void record_gr_in_time(uint8_t my_kind,
                                                uint32_t latency_us) noexcept;
    //! Record gate-return outcome: "not in time" (no CAS success
    //! before the next gate-return decision).  active_kind: the
    //! currently-dominant kind on the Linkage (= which kind's count
    //! is highest in the window — likely what peer is busy with).
    DECLSPEC_KAME static void record_gr_not_in_time(uint8_t my_kind,
                                                    uint8_t active_kind) noexcept;
    //! Record gate-return outcome: "CAS failed" (spin block broke out
    //! with WON, caller re-tried the CAS but lost).  Separates the
    //! WON-but-still-conflict case from the "WON and committed"
    //! (in_time) and "WON but no CAS happened before the next gate
    //! decision" (not_in_time) buckets.  my_kind: 1=B, 2=U.
    DECLSPEC_KAME static void record_gr_cas_fail(uint8_t my_kind) noexcept;
    //! Record the `tighten` level at the entry of `_neg_spin_block`
    //! when prev_failed was set (= previous gate-return did NOT lead
    //! to a CAS success on this Snapshot).  Level distribution maps
    //! the post-WON failure depth: most hits at level 0 → WON usually
    //! succeeds; spread toward max → repeated WON-then-fail cycles.
    DECLSPEC_KAME static void record_gr_tighten_level(uint8_t level) noexcept;
#else
    static void record_gr_in_time(uint8_t, uint32_t) noexcept {}
    static void record_gr_not_in_time(uint8_t, uint8_t) noexcept {}
    static void record_gr_cas_fail(uint8_t) noexcept {}
    static void record_gr_tighten_level(uint8_t) noexcept {}
#endif

    NegSite() = delete;
    ~NegSite() = delete;
};

inline std::unordered_map<int, NegSite::SiteState>&
NegSite::state_map() noexcept { return *tls_state_map; }
inline NegSite::SiteState*& NegSite::current_state() noexcept {
    return *tls_current_state;
}
inline bool& NegSite::last_was_gate_return() noexcept {
    return *tls_last_was_gate_return;
}
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
inline NegSite::AutoMergeStats& NegSite::auto_merge_stats() noexcept {
    return *tls_auto_merge_stats;
}
#endif

//! Livelock observation probe (always on).
//!
//! Per-thread rolling-window observer for the LIVELOCK verdict that
//! drives the fair-mode escape in negotiate_internal.  Signal: MY
//! Transaction-retry rate vs. the LINKAGE's Transaction-commit rate on
//! a rolling ≥ 10 ms window.  Emits one stderr line per window:
//!   [ll-probe] tid=... linkage=... my_tx_retry_rate=N/s
//!              tx_commit_rate=M/s ratio=X window_ms=T
//!
//! Transaction-level, NOT CAS-level: "CAS operations succeed but
//! iterate_commit keeps invalidating the whole Transaction" is the
//! pathology we care about.  my_tx_retry comes from
//! Snapshot::m_tx_retry_count, tx_commit from Linkage::m_tx_commit_count
//! (bumped in finalizeCommitment).
//!
//! Fires only at negotiate_internal entry (slow path); gate hits and
//! lottery wins stay zero-cost.  Cost over a probe-less build is one
//! uint32_t per Snapshot, one uint64_t per Linkage, and two
//! unconditional `++` statements.
//!
//! The verdict tick body (the per-priority retry-threshold logic and
//! the stderr emit) is owned by Node<XN>::NegotiationCounter::
//! livelock_probe_tx_tick — this class only carries the per-thread
//! window state and the time helper.  Non-template + function-local
//! thread_local for the same DSO-portability reasons as NegSite.
class DECLSPEC_KAME LivelockProbe {
public:
    //! Per-thread rolling window state for the LIVELOCK observer.
    struct State {
        const void *linkage_id       = nullptr;
        int64_t     t_window_us      = 0;
        uint32_t    tx_retry_window  = 0;
        uint64_t    tx_commit_window = 0;
    };
    //! TLS accessor — one slot per thread, one symbol per program.
    DECLSPEC_KAME static State& state() noexcept;
    //! steady_clock µs (wraps the std::chrono boilerplate).
    static inline int64_t now_us() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    LivelockProbe()  = delete;
    ~LivelockProbe() = delete;
};

//! Scoped wait budget — see the `currentWaitLimit` block above for what
//! is and is not promised.
//!
//! The public form takes a **duration**, which is how a driver thinks
//! ("this cycle has 1 ms; don't spend more than that waiting"), and
//! converts to the absolute limit once at construction.  That mirrors
//! `std::condition_variable::wait_for`, which is specified in terms of
//! `wait_until` for the same reason: the primitive wants an absolute
//! time, people want a duration.
//!
//! One scope covers every transaction inside it, so the natural place is
//! the top of a driver's cycle:
//! \code
//!     void MyDriver::oneCycle() {
//!         Transactional::ScopedWaitBudget wb(1000);   // 1 ms, absolute from here
//!         node.iterate_commit_while([&](Transaction<XN> &tr) -> bool {
//!             ...
//!             return Transactional::currentWaitLimit()
//!                  > Transactional::LivelockProbe::now_us();  // give up on expiry
//!         });
//!         other.iterate_commit_while(...);            // shares the same limit
//!     }
//! \endcode
//!
//! Nesting takes the **tighter** of the two limits.  A callee must not be
//! able to grant itself more time than its caller allowed, or the
//! caller's budget would mean nothing.
#if KAME_STM_WAIT_BUDGET
//!
//! **Exemption (2026-07-31, measured):** the budget may not decline the wait
//! behind a LIVE privileged peer.  Privilege is the system's completion
//! guarantee and nothing may be immune to it — a budget-expired thread that
//! barged past fair-mode became a fair-mode-immune spinner, the same disease
//! that retired STM-HIGHEST, re-invalidating a long-closure holder while
//! honest negotiators pinned behind its privilege for 12+ s (372 HANG dumps
//! vs 0 in the field-parameter harness).  The budget bounds every OTHER wait;
//! expired-lowprio stamps still unblock inside fair_mode_blocks_me, so a dead
//! holder cannot pin a budgeted thread.
class ScopedWaitBudget {
public:
    //! \param max_wait_us Duration, µs from now.  Values <= 0 arm an
    //!        already-expired budget (never wait), which is a legitimate
    //!        state, not an error.
    explicit ScopedWaitBudget(int64_t max_wait_us) noexcept
        : m_saved(currentWaitLimit()) {
        arm_(LivelockProbe::now_us() + max_wait_us);
    }
    //! Tag for the absolute-time overload.
    struct absolute_t { explicit absolute_t() = default; };
    //! Absolute variant, for the rare caller that genuinely holds a time
    //! (an external trigger, a hardware-synchronised cycle) rather than a
    //! duration.  `abs_us` is on `LivelockProbe::now_us()`'s clock.
    ScopedWaitBudget(absolute_t, int64_t abs_us) noexcept
        : m_saved(currentWaitLimit()) { arm_(abs_us); }
    ~ScopedWaitBudget() noexcept { setCurrentWaitLimit(m_saved); }
    ScopedWaitBudget(const ScopedWaitBudget &) = delete;
    ScopedWaitBudget &operator=(const ScopedWaitBudget &) = delete;
private:
    void arm_(int64_t abs_us) noexcept {
        // 0 is the "no budget" sentinel, so an armed budget must never be
        // 0.  A limit in the past is fine and means "never wait".
        if(abs_us <= 0) abs_us = 1;
        // Tighter-wins: never extend an outer budget.
        if(m_saved != 0 && m_saved < abs_us) abs_us = m_saved;
        setCurrentWaitLimit(abs_us);
    }
    int64_t m_saved;
};
#endif // KAME_STM_WAIT_BUDGET

// Portable 64-bit popcount. Visible in transaction.h so inline member
// functions (e.g. negotiate()) can use it. GCC/Clang/MSVC intrinsics.
static inline int popcount_u64(uint64_t x) noexcept {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

// Portable 64-bit count-trailing-zeros.  Undefined for x == 0 (same
// contract as __builtin_ctzll / _BitScanForward64); all call sites mask
// to a nonzero word first.
static inline int ctz_u64(uint64_t x) noexcept {
#ifdef _MSC_VER
    unsigned long i; _BitScanForward64(&i, x); return (int)i;
#else
    return __builtin_ctzll(x);
#endif
}

//! Per-Transaction observation bitset for the contention estimate.
//!
//! Accumulates distinct ProcessCounter::id values (low 16 bits of the
//! Linkage's `m_priority_state.tid` field) observed during a
//! transaction's lifetime; the popcount feeds the √C lottery / jitter
//! scaling in negotiate_internal and selects which sleep slots
//! notify_n_contenders wakes.
//!
//! TIDs are reduced mod CAPACITY (= 512) before indexing — collisions
//! are benign (they conservatively under-estimate C, which only widens
//! the jitter range; correctness is unaffected).  Non-template by
//! design: the storage layout is XN-independent and pinning it to a
//! single type lets `Snapshot::m_tid_bitset` and the
//! `NegotiationCounter::notify_n_contenders` parameter share a type
//! without template ceremony.
class TidBitset {
public:
    static constexpr int WORDS    = 8;
    static constexpr int CAPACITY = WORDS * 64;   // 512 distinct TIDs

    //! Set the bit corresponding to `tid` (reduced mod CAPACITY).
    void observe(unsigned tid) noexcept {
        unsigned idx = tid & (CAPACITY - 1);
        m_words[idx >> 6] |= 1ULL << (idx & 63);
    }
    //! Number of distinct TIDs observed (= live contender estimate C).
    int popcount() const noexcept {
        int s = 0;
        for(int i = 0; i < WORDS; ++i)
            s += popcount_u64(m_words[i]);
        return s;
    }
    //! Word access for low-level walkers (notify_n_contenders).
    uint64_t word(int i) const noexcept { return m_words[i]; }
    //! Clear all observations.  Not called on the hot path (the Tx
    //! default-ctors the bitset to zero); provided for completeness.
    void clear() noexcept {
        for(auto &w : m_words) w = 0;
    }
private:
    uint64_t m_words[WORDS] = {};
};

} //namespace Transactional

#endif /*TRANSACTION_DETAIL_H*/
