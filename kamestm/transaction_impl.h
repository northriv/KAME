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
#include "transaction.h"
#include "transaction_definitions.h"
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>

// All KAME_STM_* / KAME_LEASE_* / KAME_DT2_* / KAME_GATE_* / KAME_SPIN_*
// tuning macros live in transaction_definitions.h (-D overridable).

// STRICT_assert / STRICT_TEST — debug-build-only macros.
// STRICT_assert(expr): behaves like assert(expr) when TRANSACTIONAL_STRICT_assert is defined;
//   compiles to nothing otherwise. Used to verify STM invariants (e.g. packet consistency)
//   that are too expensive for release builds.
// STRICT_TEST(expr): evaluates expr only in strict-assert builds. Typically used to maintain
//   auxiliary state (e.g. s_serial_abandoned) consumed by STRICT_assert checks.
#ifdef TRANSACTIONAL_STRICT_assert
    #undef STRICT_assert
    #define STRICT_assert(expr) assert(expr)
    #define STRICT_TEST(expr) expr
#else
    #define STRICT_assert(expr)
    #define STRICT_TEST(expr)
#endif

namespace Transactional {

// Per-thread runner counter infrastructure. The vector and registration
// helper live in Transactional::detail (see transaction.h for design).
// Definitions are non-templated (only one Node<XN> instantiation in
// practice). transaction_impl.h is included from exactly one TU per
// binary (libkame's xnode.cpp; each tests/transaction_*_test.cpp), so
// these can be plain (non-inline) definitions. macOS two-level
// namespace would otherwise duplicate `inline` storage across
// libkame.dylib and module dylibs (libthamway, libnmr, ...) and
// break the runner-counter / nest-depth singleton invariants.
namespace detail {

// Per-thread TLS singletons — `DECLSPEC_KAME` exports each XThreadLocal
// object from libkame; plugin DLLs see the same object address via
// dllimport, which is what `detail::tls_storage()` uses as the slot key.
// (See threadlocal.h for the type-erased dispatcher design.)
DECLSPEC_KAME XThreadLocal<int, STxNestTag>     s_tx_nest;
DECLSPEC_KAME XThreadLocal<int, SSleepNestTag>  s_sleep_nest;
DECLSPEC_KAME XThreadLocal<void*, TlsPayloadCreatorPtrTag>
                                                tls_payload_creator_ptr;
DECLSPEC_KAME XThreadLocal<TlsSerial>           tls_serial;
DECLSPEC_KAME XThreadLocal<RunnerCounterEntry*, TlsRunnerCounterPtrTag>
                                                tls_runner_counter_ptr;
DECLSPEC_KAME XThreadLocal<StampKind, SCurrentOpKindTag>
                                                s_current_op_kind;
#if KAME_ENABLE_RUNNER_DIGEST
DECLSPEC_KAME XThreadLocal<RunnerDigest>        tls_runner_digest;
#endif

// =====================================================================
// Per-thread heap entries linked into a singly-linked list anchored
// at `s_runner_entries_head`.
//
// Each thread allocates its own `RunnerCounterEntry` on first STM
// use (`runner_counter_register`).  Heap allocation via `new` →
// **Linux first-touch policy places the entry on that thread's
// local NUMA node** → subsequent per-tx `fetch_add/sub` on the
// entry's `v` is a *local* atomic (~10 ns intra-socket on x86_64, ~5 ns on ARM64),
// not a cross-socket atomic (~1 µs across NUMA sockets).
//
// Slot reuse via `claimed` flag: when a thread exits, its TLS dtor
// (`RunnerEntryReleaseGuard`) sets `claimed=false`, leaving the
// entry in the list available for the next first-time-registering
// thread to CAS-claim.  This bounds memory by **max-ever-concurrent
// thread count** (not max-ever-total), handling long-running
// processes with worker spawn/destroy churn.
//
// NUMA-aware reuse (Linux only): `runner_counter_register` does a
// two-pass scan — first pass prefers entries with `allocated_node`
// matching the calling thread's current NUMA (via `getcpu(2)`),
// second pass takes any unclaimed entry.  Steady-state drives
// entries to settle on threads of matching NUMA → cross-socket
// fetch_add/sub on reused entries is bounded to brief transients.
//
// `v` encoding (simple, pre-chunked semantics):
//   v == 0 : not currently in a Tx
//   v >= 1 : in a Tx (AcquireOneCount RAII keeps v ∈ {0, 1})
//
// Replaces:
//   - the original `atomic_shared_ptr<vec<lsp<Entry>>>` design
//   - the intermediate chunked-array design
// =====================================================================

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
// Returns the NUMA node of the CPU currently scheduling this thread,
// or -1 if unknown / syscall failed.  Used by `runner_counter_register`
// to pick entries on the calling thread's local NUMA preferentially.
//
// `getcpu(2)` is fast (vDSO-accelerated on x86_64); call frequency
// is once per thread first-register, so even a plain syscall would
// be acceptable.  No `::` prefix on `syscall` — glibc declares it
// in the unistd.h namespace without making it a global-scope symbol
// reachable via `::`.
static inline int8_t kame_current_numa_node() noexcept {
#ifdef SYS_getcpu
    unsigned int cpu = 0, node = 0;
    if(syscall(SYS_getcpu, &cpu, &node, nullptr) == 0)
        return (int8_t)((node > 127) ? 127 : node);
#endif
    return -1;
}
#else
static inline int8_t kame_current_numa_node() noexcept { return -1; }
#endif

// DECLSPEC_KAME on the definition — MSVC requires it for cross-DLL
// singleton symbols (without it each module DLL gets its own copy).
DECLSPEC_KAME std::atomic<RunnerCounterEntry*> s_runner_entries_head{nullptr};

namespace {
// Process-exit cleanup: walk the entry list and delete every entry.
// Runs after main() returns, after all threads have joined; no
// concurrent access at this point.  Purely a hygiene measure so leak
// detectors (ASan, valgrind) don't complain — runtime behavior is
// unaffected because entries never shrink during operation.
struct RunnerEntriesTeardown {
    ~RunnerEntriesTeardown() noexcept {
        RunnerCounterEntry *e = s_runner_entries_head.exchange(
            nullptr, std::memory_order_acq_rel);
        while(e) {
            RunnerCounterEntry *nx =
                e->next.load(std::memory_order_relaxed);
            delete e;
            e = nx;
        }
    }
};
RunnerEntriesTeardown s_runner_entries_teardown;
} // anonymous namespace

// TLS RAII: on thread exit, mark the entry unclaimed so the next
// first-time-registering thread can pick it up via slot reuse.
// Resets `v` to 0 explicitly for safety against the (impossible
// under correct RAII, but cheap-to-defend-against) case of a thread
// exiting mid-transaction with v=1 — the next claimer needs a
// clean slate.
//
// Single relaxed-release atomic store, no allocator interaction,
// no atomic_shared_ptr, no CAS retry — safe under any TLS-
// destruction order (the issue that produced ~2-5/100 crashes in
// the prior atomic_shared_ptr<vec> design).
struct RunnerEntryReleaseGuard {
    RunnerCounterEntry *entry;
    RunnerEntryReleaseGuard() noexcept : entry(nullptr) {}
    ~RunnerEntryReleaseGuard() noexcept {
        if(entry) {
            entry->v.store(0, std::memory_order_release);
            entry->claimed.store(false, std::memory_order_release);
        }
    }
    RunnerEntryReleaseGuard(const RunnerEntryReleaseGuard &) = delete;
    RunnerEntryReleaseGuard &operator=(const RunnerEntryReleaseGuard &) = delete;
};

// Intra-libkame only — no plugin TU touches it.
static XThreadLocal<RunnerEntryReleaseGuard> tls_runner_entry_release_guard;

// Try to claim an existing unclaimed entry.  Two-pass design:
//   Pass 1 (Linux only, when my_node >= 0): prefer entries with
//          matching `allocated_node` — keeps post-reuse fetch_add
//          local to the new owner's NUMA.
//   Pass 2: take any unclaimed entry as fallback.
// Returns nullptr if all entries are claimed.
static RunnerCounterEntry* try_claim_existing_entry(int8_t my_node) noexcept {
    if(my_node >= 0) {
        for(RunnerCounterEntry *e = s_runner_entries_head.load(
                std::memory_order_acquire);
            e; e = e->next.load(std::memory_order_relaxed)) {
            if(e->allocated_node != my_node) continue;
            // Relaxed read first to avoid wasted CAS attempts on
            // entries that are visibly claimed.
            if(e->claimed.load(std::memory_order_relaxed)) continue;
            bool exp = false;
            if(e->claimed.compare_exchange_strong(
                   exp, true,
                   std::memory_order_acq_rel,
                   std::memory_order_relaxed))
                return e;
        }
    }
    for(RunnerCounterEntry *e = s_runner_entries_head.load(
            std::memory_order_acquire);
        e; e = e->next.load(std::memory_order_relaxed)) {
        if(e->claimed.load(std::memory_order_relaxed)) continue;
        bool exp = false;
        if(e->claimed.compare_exchange_strong(
               exp, true,
               std::memory_order_acq_rel,
               std::memory_order_relaxed))
            return e;
    }
    return nullptr;
}

DECLSPEC_KAME RunnerCounterEntry& runner_counter_register() {
    auto &guard = *tls_runner_entry_release_guard;
    const int8_t my_node = kame_current_numa_node();
    // First try slot reuse (bounded memory in the face of thread
    // spawn/destroy churn).
    RunnerCounterEntry *e = try_claim_existing_entry(my_node);
    if( !e) {
        // No free slot — allocate a new entry on the *calling*
        // thread's heap.  On Linux with the default first-touch
        // NUMA policy this places the page on the thread's local
        // NUMA node — making all subsequent `fetch_add/sub`
        // operations on this entry's `v` local atomics.  The
        // constructor records `allocated_node` so future reuse
        // attempts can prefer same-NUMA entries.
        e = new RunnerCounterEntry(my_node);
        // CAS-prepend onto the global list.  Set `next` BEFORE
        // the CAS; the CAS's `acq_rel` on success publishes `e`'s
        // initial state (v=0, claimed=true) and its `next` pointer
        // to any reader that subsequently does
        // `s_runner_entries_head.load(acquire)`.
        RunnerCounterEntry *head = s_runner_entries_head.load(
            std::memory_order_relaxed);
        do {
            e->next.store(head, std::memory_order_relaxed);
        } while( !s_runner_entries_head.compare_exchange_weak(
                      head, e,
                      std::memory_order_acq_rel,
                      std::memory_order_relaxed));
    }
    *tls_runner_counter_ptr = e;
    guard.entry = e;
    return *e;
}

DECLSPEC_KAME RunnerCounterEntry& my_runner_counter_impl() {
    auto *p = *tls_runner_counter_ptr;
    if(p) return *p;
    return runner_counter_register();
}

DECLSPEC_KAME unsigned int num_threads_running_impl(
    unsigned int ceiling) noexcept {
    //! Walk the per-thread entry list summing `v` values.  `v` is 0
    //! (not in Tx) or 1 (in Tx) per the AcquireOneCount semantics,
    //! so the sum directly equals the number of threads currently
    //! in a transaction.
    //!
    //! **Ceiling early-exit**: hot-path callers compare the result
    //! against very small thresholds (`KAME_STM_MAX_RUNNERS = 2` or
    //! `min_r = 1..2`); we exit as soon as the partial sum reaches
    //! `ceiling`.  Under heavy load most entries have v=1, so the
    //! loop typically touches only the first few entries.
    //!
    //! Memory ordering:
    //!   - `s_runner_entries_head.load(acquire)` synchronises with
    //!     the CAS-prepend that published any entry.  Once an entry
    //!     is in the list its `next` pointer is immutable (single-
    //!     CAS prepend at head; existing chain is never modified),
    //!     so subsequent `next.load(relaxed)` is safe.
    //!   - Per-entry `v.load(acquire)` pairs with the `release`
    //!     ordering on `AcquireOneCount` / `ReleaseOneCount` RMW,
    //!     collapsing the cross-socket NUMA visibility window from
    //!     "C++ memory model eventual" (tens of µs) to hardware
    //!     cache coherency RTT (sub-µs).  On x86 the acquire load
    //!     is a plain MOV; on ARM it compiles to LDAR.
    //!
    //! No shared refcount, no tag-CAS, no atomic_shared_ptr
    //! conversion: this is `load_acquire(head) → for each entry:
    //! load_acquire(v), load_relaxed(next)`.
    uint64_t s = 0;
    for(RunnerCounterEntry *e = s_runner_entries_head.load(
            std::memory_order_acquire);
        e; e = e->next.load(std::memory_order_relaxed)) {
        uint64_t v = e->v.load(std::memory_order_acquire);
        s += v;
        if(s >= ceiling) return ceiling;
    }
    return (unsigned)s;
}

} // namespace detail

// =============================================================================
// NegSite — per-call-site state (adaptive machine + cumulative counters).
// Declared in transaction.h; storage and impls live here (single-TU-
// per-binary include site, matching detail:: TLS above).
// =============================================================================

// NegSite TLS — class-static XThreadLocal definitions.  Inline
// accessors in transaction.h dereference these.
DECLSPEC_KAME XThreadLocal<std::unordered_map<int, NegSite::SiteState>,
                            NegSite::StateMapTag>
                                              NegSite::tls_state_map;
DECLSPEC_KAME XThreadLocal<NegSite::SiteState *, NegSite::CurrentStateTag>
                                              NegSite::tls_current_state;
DECLSPEC_KAME XThreadLocal<bool, NegSite::LastWasGateReturnTag>
                                              NegSite::tls_last_was_gate_return;

//! Global aggregator — threads merge their TLS state_map here at
//! thread exit (when INSTRUMENT wires up AutoMergeStats) or on demand
//! via mergeStatsToGlobal() / dump().  Always exists; mutex-guarded;
//! merges are rare so contention is negligible.
namespace {
struct GlobalNegStats {
    std::mutex mu;
    std::unordered_map<int, NegSite::SiteState> agg;
};
inline GlobalNegStats& globalNegStats() noexcept {
    static GlobalNegStats g;
    return g;
}
} // anonymous namespace

#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
DECLSPEC_KAME XThreadLocal<NegSite::AutoMergeStats, NegSite::AutoMergeStatsTag>
                                              NegSite::tls_auto_merge_stats;

NegSite::AutoMergeStats::~AutoMergeStats() noexcept {
    NegSite::mergeStatsToGlobal();
}

// Global Linkage-flip aggregator — 4x4 atomic matrix indexed by
// [prev_kind][curr_kind] (StampKind values 0..3), plus a log-binned
// histogram of the inter-flip interval in µs.  Incremented from
// Snapshot::tag_as_contender when a contention flip is detected on a
// Linkage.  Read+printed by NegSite::dump().
namespace {
std::atomic<uint64_t> g_linkage_flips[4][4]{};
// Log-binned histogram of inter-flip intervals (µs).  8 buckets:
//   0: <1us, 1: 1-3, 2: 3-10, 3: 10-30, 4: 30-100, 5: 100-300,
//   6: 300-1000, 7: ≥1000us
std::atomic<uint64_t> g_flip_period_hist[8]{};
inline int flip_period_bucket(uint32_t us) noexcept {
    if(us < 1)    return 0;
    if(us < 3)    return 1;
    if(us < 10)   return 2;
    if(us < 30)   return 3;
    if(us < 100)  return 4;
    if(us < 300)  return 5;
    if(us < 1000) return 6;
    return 7;
}
} // anonymous namespace

DECLSPEC_KAME void NegSite::record_linkage_flip(uint8_t prev_kind,
                                                uint8_t curr_kind,
                                                uint32_t interval_us) noexcept {
    g_linkage_flips[prev_kind & 0x3][curr_kind & 0x3]
        .fetch_add(1, std::memory_order_relaxed);
    g_flip_period_hist[flip_period_bucket(interval_us)]
        .fetch_add(1, std::memory_order_relaxed);
}

// Spin-event counters (PR3): 6 outcomes × cumulative count, plus a
// separate accumulator for the WON elapsed_us so we can compute the
// average spin duration on success.
namespace {
std::atomic<uint64_t> g_spin_count[8]{};
std::atomic<uint64_t> g_spin_won_elapsed_us_sum{0};
std::atomic<uint64_t> g_spin_timeout_elapsed_us_sum{0};
// Gate-return band counters: [kind][outcome][tighten].
// kind: 0=outer (mk=NONE) / 1=B / 2=U / 3=C
// outcome: 0=BELOW_LOW / 1=IN_BAND / 2=ABOVE_HIGH
// tighten: 0..7 (KAME_GATE_RETURN_MAX_TIGHTEN cap)
std::atomic<uint64_t> g_band_count[4][3][8]{};
// Gate-return outcome counters: in-time vs not-in-time.
//   g_gr_in_time_hist[my_kind][bucket]: latency histogram on success
//   g_gr_not_in_time[my_kind][active_kind]: who was busy when we failed
//   g_gr_cas_fail[my_kind]: WON-then-CAS-failed
//     (= spin block broke out, caller re-tried CAS, CAS lost again).
//     Distinguishes "spin won but CAS still lost" (this counter) from
//     "spin won, CAS succeeded" (g_gr_in_time_hist) and "spin won but
//     no CAS attempted before next negotiate" (g_gr_not_in_time).
std::atomic<uint64_t> g_gr_in_time_hist[4][8]{};
std::atomic<uint64_t> g_gr_not_in_time[4][4]{};
std::atomic<uint64_t> g_gr_cas_fail[4]{};
// `tighten` increments at `_neg_spin_block` entry when the previous
// gate-return for this Snapshot did not lead to a CAS success
// (snap.m_last_gate_returned was still true).  Per-level histogram
// gives the post-WON failure profile: g_gr_tighten_hist[level]
// counts how many gate-returns were observed at each tighten level
// (level 0 = first, level KAME_GATE_RETURN_MAX_TIGHTEN = saturated).
// (level i counts can be summed and compared against WON-event count
// from spin_count[WON]: when most WON events do reach a CAS success,
// almost all hits fall in level 0; persistent failure spreads the
// distribution toward the max.)
std::atomic<uint64_t> g_gr_tighten_hist[8]{};

static inline unsigned gr_latency_bucket(uint32_t us) noexcept {
    // Log2 buckets: 0=<1µs, 1=<2µs, 2=<4, 3=<8, 4=<16, 5=<32, 6=<64,
    // 7=>=64µs.
    if(us < 1)   return 0;
    if(us < 2)   return 1;
    if(us < 4)   return 2;
    if(us < 8)   return 3;
    if(us < 16)  return 4;
    if(us < 32)  return 5;
    if(us < 64)  return 6;
    return 7;
}
} // anonymous namespace

// Per-Linkage privilege diagnostic counters (KAME_ADAPT_INSTRUMENT).
// Out of the anonymous namespace so transaction_neg_impl.h (included
// later in this same TU) can reference them via `extern` linkage.
// Tracks claim attempts/successes and the priv-state at every
// `_negotiate_internal` entry.  Dumped by `NegSite::dump`.
std::atomic<uint64_t> g_neg_claim_attempts{0};
std::atomic<uint64_t> g_neg_claim_successes{0};
std::atomic<uint64_t> g_neg_internal_calls_non_priv{0};
std::atomic<uint64_t> g_neg_internal_calls_priv{0};

DECLSPEC_KAME void NegSite::record_band_event(uint8_t kind,
                                              BandOutcome outcome,
                                              uint8_t tighten) noexcept {
    const unsigned ki = kind & 0x3u;
    const unsigned oi = (unsigned)outcome & 0x3u;
    const unsigned ti = tighten & 0x7u;
    if(oi < 3)
        g_band_count[ki][oi][ti].fetch_add(1, std::memory_order_relaxed);
}

DECLSPEC_KAME void NegSite::record_gr_in_time(uint8_t my_kind,
                                              uint32_t latency_us) noexcept {
    g_gr_in_time_hist[my_kind & 0x3u][gr_latency_bucket(latency_us)]
        .fetch_add(1, std::memory_order_relaxed);
}

DECLSPEC_KAME void NegSite::record_gr_not_in_time(uint8_t my_kind,
                                                  uint8_t active_kind) noexcept {
    g_gr_not_in_time[my_kind & 0x3u][active_kind & 0x3u]
        .fetch_add(1, std::memory_order_relaxed);
}

DECLSPEC_KAME void NegSite::record_gr_cas_fail(uint8_t my_kind) noexcept {
    g_gr_cas_fail[my_kind & 0x3u]
        .fetch_add(1, std::memory_order_relaxed);
}

DECLSPEC_KAME void NegSite::record_gr_tighten_level(uint8_t level) noexcept {
    if(level < 8)
        g_gr_tighten_hist[level].fetch_add(1, std::memory_order_relaxed);
}

DECLSPEC_KAME void NegSite::record_spin_event(SpinOutcome o,
                                              uint32_t elapsed_us) noexcept {
    const unsigned idx = (unsigned)o & 0x7;
    if(idx < 8)
        g_spin_count[idx].fetch_add(1, std::memory_order_relaxed);
    if(o == SpinOutcome::WON)
        g_spin_won_elapsed_us_sum
            .fetch_add(elapsed_us, std::memory_order_relaxed);
    else if(o == SpinOutcome::TIMEOUT)
        g_spin_timeout_elapsed_us_sum
            .fetch_add(elapsed_us, std::memory_order_relaxed);
}
#endif

DECLSPEC_KAME void NegSite::mergeStatsToGlobal() noexcept {
    auto &g = globalNegStats();
    std::lock_guard<std::mutex> _lk(g.mu);
    auto &my = state_map();
    for(auto &kv : my) {
        auto &dst = g.agg[kv.first];
        // Cumulative counters: sum.
        dst.entries          += kv.second.entries;
        dst.commits          += kv.second.commits;
        for(int i = 0; i < 4; ++i) {
            dst.blocked_by_peer[i]      += kv.second.blocked_by_peer[i];
            dst.gate_returns_by_peer[i] += kv.second.gate_returns_by_peer[i];
        }
        dst.gate_then_cas_fail += kv.second.gate_then_cas_fail;
        // Adaptive-mode flip counters: sum.
        dst.mode_flips_g2n     += kv.second.mode_flips_g2n;
        dst.mode_flips_n2g     += kv.second.mode_flips_n2g;
        dst.mode_flips_promote += kv.second.mode_flips_promote;
        // Current state fields: MAX of take_gate so the "stickiest"
        // forcing across threads shows in dump (-1 < 0 < 1 →
        // FORCE_GATE dominates FORCE_SLEEP dominates UNDEFINED).
        if(kv.second.take_gate > dst.take_gate)
            dst.take_gate = kv.second.take_gate;
    }
    my.clear();
}

DECLSPEC_KAME void NegSite::dump(std::FILE *fp) noexcept {
    if( !fp) fp = stderr;
    mergeStatsToGlobal();   // pull in this thread's state first
    auto &g = globalNegStats();
    std::lock_guard<std::mutex> _lk(g.mu);
    if(g.agg.empty()) {
        std::fprintf(fp, "[neg_site stats] (empty)\n");
        return;
    }
    // Sort by call-site line ascending for stable output.
    std::vector<std::pair<int, SiteState>> rows(g.agg.begin(), g.agg.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b){ return a.first < b.first; });
    std::fprintf(fp, "[neg_site stats]\n");
    static const char *kindLabel[4] = {"N", "B", "U", "M"};
    for(auto &row : rows) {
        const int line = row.first;
        const auto &s = row.second;
        const double commit_rate = s.entries
            ? 100.0 * (double)s.commits / (double)s.entries : 0.0;
        const uint64_t gate_tot = s.gate_returns_by_peer[0]
                                + s.gate_returns_by_peer[1]
                                + s.gate_returns_by_peer[2]
                                + s.gate_returns_by_peer[3];
        const uint64_t blk_tot  = s.blocked_by_peer[0]
                                + s.blocked_by_peer[1]
                                + s.blocked_by_peer[2]
                                + s.blocked_by_peer[3];
        const double gate_then_fail_rate = gate_tot
            ? 100.0 * (double)s.gate_then_cas_fail / (double)gate_tot : 0.0;
        std::fprintf(fp,
            "  [line=%d]  entries=%llu commit_rate=%.1f%%\n",
            line,
            (unsigned long long)s.entries, commit_rate);
        std::fprintf(fp,
            "    gate_returns(%s/%s/%s/%s) = %llu / %llu / %llu / %llu  (tot=%llu)\n",
            kindLabel[0], kindLabel[1], kindLabel[2], kindLabel[3],
            (unsigned long long)s.gate_returns_by_peer[0],
            (unsigned long long)s.gate_returns_by_peer[1],
            (unsigned long long)s.gate_returns_by_peer[2],
            (unsigned long long)s.gate_returns_by_peer[3],
            (unsigned long long)gate_tot);
        std::fprintf(fp,
            "    blocked     (%s/%s/%s/%s) = %llu / %llu / %llu / %llu  (tot=%llu)\n",
            kindLabel[0], kindLabel[1], kindLabel[2], kindLabel[3],
            (unsigned long long)s.blocked_by_peer[0],
            (unsigned long long)s.blocked_by_peer[1],
            (unsigned long long)s.blocked_by_peer[2],
            (unsigned long long)s.blocked_by_peer[3],
            (unsigned long long)blk_tot);
        std::fprintf(fp,
            "    gate->cas_fail = %llu  (%.1f%% of gate_returns)\n",
            (unsigned long long)s.gate_then_cas_fail, gate_then_fail_rate);
        std::fprintf(fp,
            "    adaptive: state=%s  flips →SLEEP=%llu  →UNDEF=%llu  →GATE=%llu\n",
            (s.take_gate == -1) ? "UNDEFINED"
                : (s.take_gate == 0 ? "FORCE_SLEEP" : "FORCE_GATE"),
            (unsigned long long)s.mode_flips_g2n,
            (unsigned long long)s.mode_flips_n2g,
            (unsigned long long)s.mode_flips_promote);
    }
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    // Linkage-flip matrix — cumulative count of B/U-style kind
    // transitions across all Linkages, by (prev_kind, curr_kind).
    // Read snapshot under no lock; rows/cols labelled N/B/U/M.
    static const char *kindLabel2[4] = {"N", "B", "U", "M"};
    uint64_t flips[4][4];
    uint64_t total = 0;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j) {
            flips[i][j] = g_linkage_flips[i][j].load(std::memory_order_relaxed);
            total += flips[i][j];
        }
    std::fprintf(fp, "\n[linkage flip matrix]  (prev → curr)  total=%llu\n",
                 (unsigned long long)total);
    std::fprintf(fp, "                curr=N         B         U         M\n");
    for(int i = 0; i < 4; ++i) {
        std::fprintf(fp, "  prev=%s   ", kindLabel2[i]);
        for(int j = 0; j < 4; ++j) {
            std::fprintf(fp, " %9llu",
                         (unsigned long long)flips[i][j]);
        }
        std::fprintf(fp, "\n");
    }
    // Per-pair B/U cycle and B/M cycle detection — the most directly
    // wasteful patterns (bundle-then-unbundle on same parent).
    if(total > 0) {
        uint64_t bu = flips[1][2] + flips[2][1];  // B↔U cycling
        uint64_t bm = flips[1][3] + flips[3][1];  // B↔M cycling
        std::fprintf(fp,
            "  B↔U flips = %llu (%.1f%%)   B↔M flips = %llu (%.1f%%)\n",
            (unsigned long long)bu, 100.0 * (double)bu / (double)total,
            (unsigned long long)bm, 100.0 * (double)bm / (double)total);
    }
    // Period histogram (inter-flip interval, log-binned in µs).
    // Buckets:  <1us, 1-3, 3-10, 10-30, 30-100, 100-300, 300-1000, ≥1000us
    uint64_t hist[8];
    uint64_t hist_total = 0;
    for(int i = 0; i < 8; ++i) {
        hist[i] = g_flip_period_hist[i].load(std::memory_order_relaxed);
        hist_total += hist[i];
    }
    if(hist_total > 0) {
        static const char *bucketLabel[8] = {
            "<1us", "1-3us", "3-10us", "10-30us",
            "30-100us", "100-300us", "300us-1ms", ">=1ms"
        };
        std::fprintf(fp, "[linkage flip period histogram]  samples=%llu\n",
                     (unsigned long long)hist_total);
        // Compute weighted median bucket (for a quick "typical period" read).
        uint64_t acc = 0;
        int median_bucket = 0;
        for(int i = 0; i < 8; ++i) {
            acc += hist[i];
            if(acc * 2 >= hist_total) { median_bucket = i; break; }
        }
        for(int i = 0; i < 8; ++i) {
            double pct = 100.0 * (double)hist[i] / (double)hist_total;
            std::fprintf(fp, "  %-10s : %9llu (%5.1f%%)%s\n",
                         bucketLabel[i],
                         (unsigned long long)hist[i], pct,
                         (i == median_bucket) ? "  ← median" : "");
        }
    }
    // Spin-for-same-kind outcomes.
    uint64_t spin_counts[8];
    uint64_t spin_total = 0, spin_attempts = 0;
    for(int i = 0; i < 8; ++i) {
        spin_counts[i] = g_spin_count[i].load(std::memory_order_relaxed);
        spin_total += spin_counts[i];
    }
    spin_attempts = spin_counts[4] + spin_counts[5];   // WON + TIMEOUT
    if(spin_total > 0) {
        static const char *spinLabel[8] = {
            "skipped(no_period)", "skipped(cold)", "skipped(past)",
            "skipped(same_kind)", "won", "timeout",
            "skipped(thrashing)", "gate_return(same_kind)"
        };
        std::fprintf(fp, "[spin-for-same-kind]  total_calls=%llu attempts=%llu\n",
                     (unsigned long long)spin_total,
                     (unsigned long long)spin_attempts);
        for(int i = 0; i < 8; ++i) {
            double pct = 100.0 * (double)spin_counts[i] / (double)spin_total;
            std::fprintf(fp, "  %-20s : %9llu (%5.1f%%)\n",
                         spinLabel[i],
                         (unsigned long long)spin_counts[i], pct);
        }
        if(spin_attempts > 0) {
            uint64_t won = spin_counts[4];
            uint64_t tmo = spin_counts[5];
            uint64_t won_us = g_spin_won_elapsed_us_sum
                                  .load(std::memory_order_relaxed);
            uint64_t tmo_us = g_spin_timeout_elapsed_us_sum
                                  .load(std::memory_order_relaxed);
            double win_rate = 100.0 * (double)won / (double)spin_attempts;
            std::fprintf(fp, "  win_rate = %.1f%% (%llu/%llu)\n",
                         win_rate,
                         (unsigned long long)won,
                         (unsigned long long)spin_attempts);
            if(won > 0)
                std::fprintf(fp, "  avg_won_elapsed     = %.1f us\n",
                             (double)won_us / (double)won);
            if(tmo > 0)
                std::fprintf(fp, "  avg_timeout_elapsed = %.1f us\n",
                             (double)tmo_us / (double)tmo);
        }
    }
    // Gate-return diagnostics.
    static const char *grKindLabel[4] = {"outer", "B", "U", "C"};
    static const char *bandLabel[3] = {"below_low", "in_band", "above_hi"};
    bool any_band = false;
    for(int k = 0; k < 4 && !any_band; ++k)
        for(int b = 0; b < 3 && !any_band; ++b)
            for(int t = 0; t < 8 && !any_band; ++t)
                if(g_band_count[k][b][t].load(std::memory_order_relaxed))
                    any_band = true;
    if(any_band) {
        std::fprintf(fp, "[gate-return band]\n");
        for(int k = 0; k < 4; ++k) {
            for(int b = 0; b < 3; ++b) {
                uint64_t sum = 0;
                for(int t = 0; t < 8; ++t)
                    sum += g_band_count[k][b][t].load(std::memory_order_relaxed);
                if(sum == 0) continue;
                std::fprintf(fp, "  %-6s %-10s : total=%9llu",
                             grKindLabel[k], bandLabel[b],
                             (unsigned long long)sum);
                for(int t = 0; t < 8; ++t) {
                    uint64_t c = g_band_count[k][b][t]
                        .load(std::memory_order_relaxed);
                    if(c > 0)
                        std::fprintf(fp, " L%d=%llu", t,
                                     (unsigned long long)c);
                }
                std::fprintf(fp, "\n");
            }
        }
    }
    // Gate-return in-time (post-gate-return CAS success) latency.
    bool any_in_time = false;
    for(int k = 0; k < 4 && !any_in_time; ++k)
        for(int b = 0; b < 8 && !any_in_time; ++b)
            if(g_gr_in_time_hist[k][b].load(std::memory_order_relaxed))
                any_in_time = true;
    if(any_in_time) {
        static const char *latLabel[8] = {
            "<1us", "<2us", "<4us", "<8us",
            "<16us", "<32us", "<64us", ">=64us"
        };
        std::fprintf(fp, "[gate-return in-time latency]\n");
        for(int k = 0; k < 4; ++k) {
            uint64_t total = 0;
            for(int b = 0; b < 8; ++b)
                total += g_gr_in_time_hist[k][b]
                    .load(std::memory_order_relaxed);
            if(total == 0) continue;
            std::fprintf(fp, "  %-6s : total=%9llu",
                         grKindLabel[k], (unsigned long long)total);
            for(int b = 0; b < 8; ++b) {
                uint64_t c = g_gr_in_time_hist[k][b]
                    .load(std::memory_order_relaxed);
                if(c > 0)
                    std::fprintf(fp, " %s=%llu", latLabel[b],
                                 (unsigned long long)c);
            }
            std::fprintf(fp, "\n");
        }
    }
    // Gate-return not-in-time: who was active when we failed.
    bool any_nit = false;
    for(int k = 0; k < 4 && !any_nit; ++k)
        for(int a = 0; a < 4 && !any_nit; ++a)
            if(g_gr_not_in_time[k][a].load(std::memory_order_relaxed))
                any_nit = true;
    if(any_nit) {
        std::fprintf(fp, "[gate-return NOT-in-time: my_kind → active]\n");
        for(int k = 0; k < 4; ++k) {
            uint64_t total = 0;
            for(int a = 0; a < 4; ++a)
                total += g_gr_not_in_time[k][a]
                    .load(std::memory_order_relaxed);
            if(total == 0) continue;
            std::fprintf(fp, "  my=%-6s : total=%9llu",
                         grKindLabel[k], (unsigned long long)total);
            for(int a = 0; a < 4; ++a) {
                uint64_t c = g_gr_not_in_time[k][a]
                    .load(std::memory_order_relaxed);
                if(c > 0)
                    std::fprintf(fp, " active=%s:%llu", grKindLabel[a],
                                 (unsigned long long)c);
            }
            std::fprintf(fp, "\n");
        }
    }
    // Gate-return WON-then-CAS-failed: spin caught peer activity,
    // broke out, the caller's retry CAS still lost.  Pair these with
    // [gate-return in-time latency] (= WON-then-CAS-succeeded) to
    // get the WON outcome split:
    //   WON → in_time : caller's CAS won shortly after gate-return.
    //   WON → cas_fail: caller's CAS lost (peer beat us).
    //   WON → not_in_time : no CAS happened between gate-return and
    //                       the next negotiate (e.g. caller went off
    //                       to do something else, or the spin block
    //                       fired twice in a row without an
    //                       intervening CAS).
    bool any_cas_fail = false;
    for(int k = 0; k < 4 && !any_cas_fail; ++k)
        if(g_gr_cas_fail[k].load(std::memory_order_relaxed))
            any_cas_fail = true;
    if(any_cas_fail) {
        std::fprintf(fp, "[gate-return WON then CAS failed]\n");
        for(int k = 0; k < 4; ++k) {
            uint64_t c = g_gr_cas_fail[k]
                .load(std::memory_order_relaxed);
            if(c == 0) continue;
            std::fprintf(fp, "  my=%-6s : count=%llu\n",
                         grKindLabel[k], (unsigned long long)c);
        }
    }
    // Tighten-level histogram: how deep did the post-WON failure
    // ladder climb?  Sum across levels = total prev_failed events
    // observed at `_neg_spin_block` entry (= count of WON events
    // whose follow-up CAS did NOT commit by the next entry).  Most
    // hits at level 0 → WON usually leads to commit before the
    // next negotiate fires; spread toward MAX = repeated failure.
    {
        uint64_t tighten_total = 0;
        for(int i = 0; i < 8; ++i)
            tighten_total += g_gr_tighten_hist[i]
                .load(std::memory_order_relaxed);
        if(tighten_total > 0) {
            std::fprintf(fp,
                "[gate-return tighten level distribution]  total=%llu\n",
                (unsigned long long)tighten_total);
            for(int i = 0; i < 8; ++i) {
                uint64_t c = g_gr_tighten_hist[i]
                    .load(std::memory_order_relaxed);
                if(c == 0) continue;
                double pct = 100.0 * (double)c / (double)tighten_total;
                std::fprintf(fp, "  L=%d : %9llu (%5.1f%%)\n",
                             i, (unsigned long long)c, pct);
            }
        }
    }
    // Per-Linkage privilege diagnostic counters.
    {
        uint64_t cl_att = g_neg_claim_attempts.load(std::memory_order_relaxed);
        uint64_t cl_suc = g_neg_claim_successes.load(std::memory_order_relaxed);
        uint64_t ni_np  = g_neg_internal_calls_non_priv.load(std::memory_order_relaxed);
        uint64_t ni_p   = g_neg_internal_calls_priv.load(std::memory_order_relaxed);
        std::fprintf(fp,
            "[per-Linkage privilege]\n"
            "  claim attempts          : %llu\n"
            "  claim successes         : %llu  (%.1f%%)\n"
            "  _negotiate_internal     : non-priv=%llu  priv=%llu\n",
            (unsigned long long)cl_att,
            (unsigned long long)cl_suc,
            cl_att > 0 ? 100.0 * (double)cl_suc / (double)cl_att : 0.0,
            (unsigned long long)ni_np,
            (unsigned long long)ni_p);
    }
#endif
    std::fflush(fp);
}

// tx_nest / sleep_nest are defined at namespace scope in transaction.h
// (detail::s_tx_nest, detail::s_sleep_nest). Placing them outside the
// class template avoids an Apple clang / arm64 bug where the TLS wrapper
// for a template static `thread_local` member is not emitted in TUs
// that only include transaction.h.

STRICT_TEST(static atomic<int64_t> s_serial_abandoned = -2);

// (popcount_u64 moved to transaction.h; retry_pause + effective_runners
//  moved to transaction_negotiation.h.)

// LivelockProbe TLS state — non-template, one slot per thread per
// program (libkame defines the accessor once; modules link to it).
// Class declaration lives in transaction.h.
DECLSPEC_KAME LivelockProbe::State& LivelockProbe::state() noexcept {
    thread_local State v;
    return v;
}

namespace detail {

    // Fires the livelock probe. The caller (negotiate_internal, which
    // has access to Snapshot's protected members via enclosing-class
    // friendship) pre-computes the tag-ownership counts, the
    // retry-threshold (chosen from Priority), and the priority label,
    // then passes plain values here.
    //
    // Verdict:
    //   tags_owned == tags_total > 0                  AND
    //   my_tx_retries >= retry_threshold              AND
    //   tx_age_us > KAME_STM_LIVELOCK_MIN_AGE_US (20 ms default)
    //
    // retry_threshold is priority-dependent:
    //   HIGHEST         → 2  (real-time, must not retry)
    //   NORMAL          → 3
    //   UI_DEFERRABLE   → 5  (deferred UI repaint — retries tolerated)
    //   LOWEST          → 5  (background — tolerates yielding)
    //
    // Two unrelated time scales appear in the printout:
    //   tx_age_us  = age of the current Tx/Snapshot (set by Snapshot or
    //                Transaction ctor; persists across retries — what
    //                the verdict gates on).
    //   window_ms  = inter-tick interval on this (linkage, thread) —
    //                the rate-window for my_tx_retry_rate /
    //                tx_commit_rate. NOT Tx-related; in tight retry
    //                loops it is often sub-ms, which is why the printed
    //                rates can look enormous (small delta divided by a
    //                tiny window).
    //! Returns true iff a tick concluded `verdict=LIVELOCK`. Defined
    //! out-of-class as `Node<XN>::NegotiationCounter::livelock_probe_tx_tick`.

//=============================================================================
    // Fair-mode escape: globally registered "privileged TID" mechanism.
    //
    // When the livelock probe (above) returns verdict=LIVELOCK on a
    // Transaction that is the Greedy CM winner (oldest started_time,
    // all linkage tags claimed) yet making no commit progress, that Tx
    // CAS-claims `s_privileged_tidstamp`. While the slot is non-zero AND
    // != my_tid:
    //
    //   * the jittered gate and √C lottery in negotiate_internal's
    //     main sleep loop are bypassed — strict Greedy CM (older
    //     started_time wins → others sleep) is the sole priority
    //     mechanism, ensuring the privileged Tx makes progress
    //     against deterministic backoff alone.
    //   * notify_n_contenders wakes the privileged TID's sleep slot
    //     first, before walking tid_bitset.
    //
    // Per-Transaction ownership lives on `Snapshot::m_registered_privileged`
    // for nesting safety (see transaction.h). The slot is cleared by:
    //   - finalizeCommitment (success): plain store(0, relaxed)
    //   - ~Transaction (abort): same plain store
    // — only the registering Tx holds the flag, no concurrent writer
    // is possible, so a CAS would be overkill.
    //
    // V0 (legacy non-adaptive) and the V0↔ADAPTIVE mode switch were
    // removed in this revision: empirical ARM64 vs x86_64 comparison
    // (paper §3.6) showed V0 to be at best on par and sometimes 5×
    // slower than the always-on adaptive path. The fair-mode
    // escape, in contrast, is orthogonal to mode and the retry-path
    // tag_as_contender call sites are now unconditional.
    //=============================================================================

} // namespace detail
} // namespace Transactional

// ScopedNegotiateLinkage<XN> RAII + retry_pause + effective_runners
// forward decl live in transaction_negotiation.h.  WalkUpResult,
// NegotiationCounter template member bodies, negotiate_after_retry_pause
// and negotiate_internal() live in transaction_neg_impl.h.
#include "transaction_negotiation.h"
#include "transaction_neg_impl.h"

namespace Transactional {

template <class XN>
XThreadLocal<typename Node<XN>::FuncPayloadCreator> Node<XN>::stl_funcPayloadCreator;

atomic<ProcessCounter::cnt_t> ProcessCounter::s_count = ProcessCounter::MAINTHREADID - 1;
XThreadLocal<ProcessCounter> ProcessCounter::stl_processID;

ProcessCounter::ProcessCounter() noexcept {
    for(;;) {
        cnt_t oldv = s_count;
        cnt_t newv = oldv + (cnt_t)1u;
        if( !newv) ++newv;
        if(s_count.compare_set_strong(oldv, newv)) {
            //avoids zero.
//            fprintf(stderr, "Assigning a new process ID=%d\n", newv);
            m_var = newv;
            break;
        }
    }
}
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// Non-inline definition so the body compiles into kame.dll and is imported by
// plugin DLLs — ensuring *stl_processID is always kame.dll's TLS slot.
ProcessCounter::cnt_t ProcessCounter::id() noexcept { return *stl_processID; }
#endif

XThreadLocal<Priority> stl_currentPriority;

Priority getCurrentPriorityMode() {
    return *stl_currentPriority;
}

template <class XN>
void
Node<XN>::Packet::print_() const {
    printf("Packet: ");
    printf("%s@%p, ", typeid(*this).name(), &node());
    printf("BP@%p, ", node().m_link.get());
    if(missing())
        printf("missing, ");
    if(size()) {
        printf("%d subnodes : [ \n", (int)size());
        for(int i = 0; i < size(); i++) {
            if(subpackets()->at(i)) {
                subpackets()->at(i)->print_();
                printf("; ");
            }
            else {
                printf("%s@%p, w/o packet, ", typeid(*this).name(), subnodes()->at(i).get());
            }
        }
        printf("]\n");
    }
    printf(";");
}

template <class XN>
bool
Node<XN>::Packet::checkConsistensy(const local_shared_ptr<Packet> &rootpacket,
                                   const local_shared_ptr<Packet> &globalroot) const {
    // `rootpacket` switches on recursion (local sub-bundle root) for
    // the "sub missing → self missing" propagation; `groot` stays
    // unchanged through recursion and drives the Null-slot
    // reverseLookup (needed for hard-link Case B).  Default
    // `globalroot={}` degenerates groot to rootpacket — semantics doc
    // in `transaction.h`.
    const local_shared_ptr<Packet> &groot = globalroot ? globalroot : rootpacket;
    try {
        if(size()) {
            if( !(payload()->m_serial - subpackets()->m_serial < 0x7fffffffffffffffLL))
                throw __LINE__;
        }
        for(int i = 0; i < size(); i++) {
            if( !subpackets()->at(i)) {
                if( !groot->missing()) {
                    if( !subnodes()->at(i)->reverseLookup(
                        const_cast<local_shared_ptr<Packet>&>(groot), false, 0, false, 0))
                        throw __LINE__;
                }
            }
            else {
                if(subpackets()->at(i)->size())
                    if( !(subpackets()->m_serial - subpackets()->at(i)->subpackets()->m_serial < 0x7fffffffffffffffLL))
                        throw __LINE__;
                if(subpackets()->at(i)->missing() && (rootpacket.get() != this)) {
                    if( !missing())
                        throw __LINE__;
                }
                // Recurse with groot UNCHANGED (always the original
                // top-level root) — only the local root switches.
                if( !subpackets()->at(i)->checkConsistensy(
                    subpackets()->at(i)->missing() ? rootpacket : subpackets()->at(i),
                    groot))
                    return false;
            }
        }
    }
    catch (int &line) {
        fprintf(stderr, "Line %d, losing consistensy on node %p:\n", line, &node());
        groot->print_();
        throw *this;
    }
    return true;
}

template <class XN>
bool
Node<XN>::Packet::allSubReachable(const local_shared_ptr<Packet> &rootpacket,
                                  const local_shared_ptr<Packet> &globalroot) const {
    // Non-throwing mirror of checkConsistensy's Null-slot path.  Used
    // by bundle Phase 4 to gate the `is_bundle_root` `m_missing=false`
    // publish.  `globalroot` follows checkConsistensy's semantics.
    const local_shared_ptr<Packet> &groot = globalroot ? globalroot : rootpacket;
    if(groot->missing()) return true;  // root missing → no Null-slot check fires
    for(int i = 0; i < size(); i++) {
        if( !subpackets()->at(i)) [[unlikely]] {  // hard-link only
            if( !subnodes()->at(i)->reverseLookup(
                const_cast<local_shared_ptr<Packet>&>(groot), false, 0, false, 0))
                return false;
        }
        else {
            if(subpackets()->at(i)->size())
                if( !subpackets()->at(i)->allSubReachable(
                    subpackets()->at(i)->missing() ? rootpacket : subpackets()->at(i),
                    groot))
                    return false;
        }
    }
    return true;
}

template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial) noexcept :
    m_bundledBy(), m_packet(x), m_reverse_index((int)PACKET_STATE::PACKET_HAS_PRIORITY),
    m_bundle_serial(bundle_serial) {
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Linkage > &bp, int reverse_index,
    int64_t bundle_serial) noexcept :
    m_bundledBy(bp), m_packet(), m_reverse_index(),
    m_bundle_serial(bundle_serial) {
    setReverseIndex(reverse_index);
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const PacketWrapper &x, int64_t bundle_serial) noexcept :
    m_bundledBy(x.m_bundledBy), m_packet(x.m_packet),
    m_reverse_index(x.m_reverse_index), m_bundle_serial(bundle_serial) {}

template <class XN>
void
Node<XN>::PacketWrapper::print_() const {
    printf("PacketWrapper: ");
    if( !hasPriority()) {
        printf("referred to BP@%p, ", bundledBy().get());
    }
    printf("serial:%lld, ", (long long)m_bundle_serial);
    if(packet()) {
        packet()->print_();
    }
    else {
        printf("absent, ");
    }
    printf("\n");
}

template <class XN>
void
Node<XN>::print_recoverable_error(const char* reason) {
#ifdef gErrPrint
    try {
    char buf[256] = {};
        snprintf(buf, sizeof(buf), "Out of memory!: %s\nClose unnescessary windows & Store your data immediately.\n", reason);
        gErrPrint(buf);
    }
    catch (const std::bad_alloc &) {
#else
    {
#endif
        fprintf(stderr, "Memory allocation has failed: %s\nTransaction is delaying...\n", reason);
    }
    msecsleep(1000);
}


template <class XN>
Node<XN>::Node() : m_link(make_local_shared<Linkage>()) {
    local_shared_ptr<Packet> packet(make_local_shared<Packet>());
    *m_link = make_local_shared<PacketWrapper>(packet, SerialGenerator::gen());
    //Use create() for this hack.
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // Read and clear the shared void* slot — see create() for why we use
    // this instead of stl_funcPayloadCreator on Windows.
    auto creator = reinterpret_cast<FuncPayloadCreator>(*detail::tls_payload_creator_ptr);
    *detail::tls_payload_creator_ptr = nullptr;
#else
    auto creator = *stl_funcPayloadCreator;
    *stl_funcPayloadCreator = nullptr;
#endif
    packet->m_payload = creator(static_cast<XN&>( *this));
}
template <class XN>
Node<XN>::~Node() {
    releaseAll();
}
template <class XN>
void
Node<XN>::print_() const {
    scoped_atomic_view<PacketWrapper> packet( *m_link);
//	printf("Node:%p, ", this);
//	printf("BP:%p, ", m_link.get());
//	printf(" packet: ");
    packet->print_();
}

template <class XN>
void
Node<XN>::insert(const shared_ptr<XN> &var) {
    iterate_commit_if([this, var](Transaction<XN> &tr)->bool {
        return insert(tr, var);
    });
}
//=============================================================================
// insert() — add a child node to the tree within a transaction
//   (Comments by Claude Opus — based on source code analysis)
//
// When online_after_insertion is true, the child is committed to the live tree
// immediately so that tr[*child] is accessible within the same transaction.
// This requires an intermediate commit of the parent, followed by a CAS on
// the child's Linkage to point it into the parent's packet.
// Without online_after_insertion, the child becomes visible only after the
// enclosing transaction commits.
//=============================================================================
template <class XN>
bool
Node<XN>::insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    packet->subpackets() = packet->size() ? std::make_shared<PacketList>( *packet->subpackets()) : std::make_shared<PacketList>();
    packet->subpackets()->m_serial = tr.m_serial;
    packet->m_missing = true;
    packet->subnodes() = packet->size() ? std::make_shared<NodeList>( *packet->subnodes()) : std::make_shared<NodeList>();
//    if( !packet->subpackets()->size()) {
//        packet->subpackets()->reserve(4);
//        packet->subnodes()->reserve(4);
//    }
    packet->subpackets()->resize(packet->size() + 1);
    assert(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
    packet->subnodes()->resize(packet->subpackets()->size());
    packet->subnodes()->back() = var;

    if(online_after_insertion) {
        bool has_failed = false;
        //Tags serial.
        local_shared_ptr<Packet> newpacket(tr.m_packet);
        tr.m_packet = make_local_shared<Packet>( *tr.m_oldpacket);
        if( !tr.m_packet->node().commit(tr)) {
            printf("*\n");
            has_failed = true;
        }
        tr.m_oldpacket = tr.m_packet;
        tr.m_packet = newpacket;
        for(int iter = 0;; ++iter) {
            // RAII OnExit: tags only on scope exit (continue / return)
            // unless commit() is called. iter==0 successful CAS stays
            // untagged (cheap fast path).
            ScopedNegotiateLinkage<XN> scope(var->m_link, tr, iter,
                ScopedNegotiateLinkage<XN>::TagMode::OnExit);
            if( !scope) continue;  // weak acquire lost — treat as CAS failure

            local_shared_ptr<Packet> subpacket_new;
            // Pass scope directly — bundle_subpacket uses its view for
            // unbundle/bundle and updates via set_view on success.
            BundledStatus status = bundle_subpacket(0, var, scope, subpacket_new,
                tr, tr.m_serial);
            if(status != BundledStatus::SUCCESS) {
                continue;
            }
            if( !subpacket_new) {
                scope.commit();
                break;
            }

            //Marks for writing at subnode.
            tr.m_packet = make_local_shared<Packet>( *tr.m_oldpacket);
            if( !tr.m_packet->node().commit(tr)) {
                printf("&\n");
                has_failed = true;
            }
            tr.m_oldpacket = tr.m_packet;
            tr.m_packet = newpacket;

            auto newwrapper = make_local_shared<PacketWrapper>(
                m_link, packet->size() - 1, tr.m_serial);
            newwrapper->packet() = subpacket_new;
            packet->subpackets()->back() = subpacket_new;
            if(has_failed)
                return false;
            if( !scope.compareAndSetWithHint(newwrapper)) {
                tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket);
                return false;
            }
            break;
        }
    }
    tr[ *this].catchEvent(var, packet->size() - 1);
    tr[ *this].listChangeEvent();
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
//		printf("i");
}
template <class XN>
void
Node<XN>::release(const shared_ptr<XN> &var) {
    iterate_commit_if([this, var](Transaction<XN> &tr)->bool {
        return release(tr, var);
    });
}

// eraseSerials() — recursively reset all serial markers matching the given
// serial back to SERIAL_NULL. Called during release() to prevent a released
// sub-tree's stale serial from causing false collision detection in future
// transactions.
template <class XN>
void
Node<XN>::eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial,
                       Snapshot<XN> &snap) {
    if(packet->size() && packet->subpackets()->m_serial == serial)
        packet->subpackets()->m_serial = SerialGenerator::SERIAL_NULL;
    if(packet->payload()->m_serial == serial)
        packet->payload()->m_serial = SerialGenerator::SERIAL_NULL;

    for(int iter = 0;; ++iter) {
        // RAII OnExit.
        ScopedNegotiateLinkage<XN> scope(packet->node().m_link, snap, iter,
            ScopedNegotiateLinkage<XN>::TagMode::OnExit);
        if( !scope) continue;  // weak acquire lost — treat as CAS failure
        // Use scope's internal m_view directly — no separate
        // scoped_atomic_view on the same linkage needed.  scope-> reads
        // through m_view; scope.compareAndSet(newwrapper) (1-arg) uses
        // m_view as the CAS oldr.
        if(scope->m_bundle_serial != serial) {
            scope.commit();
            break;
        }
        // unique_ptr: ownership to m_link on success.
        auto newwrapper = make_local_shared<PacketWrapper>( *scope, SerialGenerator::SERIAL_NULL);
        if(scope.compareAndSet(newwrapper))
            break;
        // RAII tags on continue (iter > 0)
    }
    for(int i = 0; i < packet->size(); ++i) {
        local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
        if(subpacket)
            eraseSerials(subpacket, serial, snap);
    }
}

template <class XN>
void
Node<XN>::lookupFailure() const {
    fprintf(stderr, "Node not found during a lookup.\n");
    throw NodeNotFoundError("Lookup failure.");
}

template <class XN>
bool
Node<XN>::release(Transaction<XN> &tr, const shared_ptr<XN> &var) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    assert(packet->size());
    packet->subpackets().reset(new PacketList( *packet->subpackets()));
    packet->subpackets()->m_serial = tr.m_serial;
    packet->subnodes().reset(new NodeList( *packet->subnodes()));
    unsigned int idx = 0;
    int old_idx = -1;
    // Tag-ref view loaded once below; read for hasPriority/bundledBy
    // checks, then moved into ScopedNeg at function tail (zero extra
    // atomic ops — the scoped_atomic_view&& ctor transfers directly).
    scoped_atomic_view<PacketWrapper> nullsubwrapper;
    // newsubwrapper: ownership transferred to var->m_link on CAS
    // success.  unique_ptr saves 2 atomic ops vs local_shared_ptr.
    local_shared_ptr<PacketWrapper> newsubwrapper;
    auto nit = packet->subnodes()->begin();
    for(auto pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
        assert(nit != packet->subnodes()->end());
        if(nit->get() == &*var) {
            if( *pit) {
                nullsubwrapper = scoped_atomic_view<PacketWrapper>(
                    *var->m_link,
                    scoped_atomic_view<PacketWrapper>::ADAPTIVE_THRESHOLD,
                    /*weakly=*/true);
                if(!nullsubwrapper.acquire_succeeded()) {
                    tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket); //Following commitment should fail.
                    return false;
                }
                if(nullsubwrapper->hasPriority()) {
                    if(nullsubwrapper->packet() != *pit) {
                        tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket); //Following commitment should fail.
                        return false;
                    }
                }
                else {
                    local_shared_ptr<Linkage> bp(nullsubwrapper->bundledBy());
                    if((bp && (bp != m_link)) ||
                        ( !bp && (nullsubwrapper->packet() != *pit))) {
                        tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket); //Following commitment should fail.
                        return false;
                    }
                }
                newsubwrapper = make_local_shared<PacketWrapper>(m_link, idx, SerialGenerator::SERIAL_NULL);
                newsubwrapper->packet() = *pit;
            }
            pit = packet->subpackets()->erase(pit);
            nit = packet->subnodes()->erase(nit);
            old_idx = idx;
        }
        else {
            ++nit;
            ++pit;
            ++idx;
        }
    }
    if(old_idx < 0)
        lookupFailure();

    if( !packet->subpackets()->size()) {
        packet->subpackets().reset();
        packet->m_missing = false;
    }
    else {
//        if(packet->subpackets()->capacity() - packet->subpackets()->size() > 8) {
//            packet->subpackets()->shrink_to_fit();
//            packet->subnodes()->shrink_to_fit();
//        }
    }
    if(tr.m_packet->size()) {
        tr.m_packet->m_missing = true;
    }

    tr[ *this].releaseEvent(var, old_idx);
    tr[ *this].listChangeEvent();

    if( !newsubwrapper) {
        //Packet of the released node is held by the other point inside the tr.m_packet.
        return true;
    }

    eraseSerials(packet, tr.m_serial, tr);

    local_shared_ptr<Packet> newpacket(tr.m_packet);
    tr.m_packet = tr.m_oldpacket;
    tr.m_packet = make_local_shared<Packet>( *tr.m_packet);
    if( !tr.m_packet->node().commit(tr)) {
        tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket); //Following commitment should fail.
        tr.m_packet = newpacket;
        return false;
    }
    tr.m_oldpacket = tr.m_packet;
    tr.m_packet = newpacket;

    //Unload the packet of the released node.
    // Move the pre-loaded view directly into scope (0 atomic ops).
    ScopedNegotiateLinkage<XN> scope(var->m_link, tr, -1,
        std::move(nullsubwrapper),
        ScopedNegotiateLinkage<XN>::TagMode::OnExit);
    if( !scope.compareAndSetWithHint(newsubwrapper)) {
        tr.m_oldpacket = make_local_shared<Packet>( *tr.m_oldpacket); //Following commitment should fail.
        return false; // destructor tags
    }
    // scope auto-committed + tagged successful_cas via the call.
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
}
template <class XN>
void
Node<XN>::releaseAll() {
    iterate_commit_if([this](Transaction<XN> &tr)->bool {
        while(tr.size()) {
            if( !release(tr, tr.list()->back())) {
                return false;
            }
        }
        return true;
    });
}
template <class XN>
void
Node<XN>::swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
    iterate_commit_if([this, x, y](Transaction<XN> &tr)->bool {
        return swap(tr, x, y);
    });
}
template <class XN>
bool
Node<XN>::swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
    packet->subpackets()->m_serial = tr.m_serial;
    packet->m_missing = true;
    packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
    unsigned int idx = 0;
    int x_idx = -1, y_idx = -1;
    for(auto nit = packet->subnodes()->begin(); nit != packet->subnodes()->end(); ++nit) {
        if( *nit == x)
            x_idx = idx;
        if( *nit == y)
            y_idx = idx;
        ++idx;
    }
    if((x_idx < 0) || (y_idx < 0))
        lookupFailure();
    local_shared_ptr<Packet> px = packet->subpackets()->at(x_idx);
    local_shared_ptr<Packet> py = packet->subpackets()->at(y_idx);
    packet->subpackets()->at(x_idx) = py;
    packet->subpackets()->at(y_idx) = px;
    packet->subnodes()->at(x_idx) = y;
    packet->subnodes()->at(y_idx) = x;
    tr[ *this].moveEvent(x_idx, y_idx);
    tr[ *this].listChangeEvent();
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
}

// reverseLookupWithHint() — fast path for finding a node's packet within a
// transaction's packet tree. Uses the bundledBy back-reference chain stored
// in the node's PacketWrapper to walk directly to the containing slot,
// avoiding a full tree scan. Falls back to nullptr if the hint is stale.
// When copy_branch is true, applies copy-on-write along the path.
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(local_shared_ptr<Linkage> &linkage,
    local_shared_ptr<Packet> &superpacket, bool copy_branch, int64_t tr_serial, bool set_missing,
    local_shared_ptr<Packet> *upperpacket, int *index) {
    if( !superpacket->size())
        return nullptr;
    // Tag-ref view instead of load_shared_ — avoids fetch_add/sub pair
    // for this short-lived read (hasPriority, bundledBy, reverseIndex).
    scoped_atomic_view<PacketWrapper> wrapper( *linkage);
    if( !wrapper) return nullptr;
    if(wrapper->hasPriority())
        return nullptr;
    local_shared_ptr<Linkage> linkage_upper(wrapper->bundledBy());
    if( !linkage_upper)
        return nullptr;
    local_shared_ptr<Packet> *foundpacket;
    if(linkage_upper == superpacket->node().m_link)
        foundpacket = &superpacket;
    else {
        foundpacket = reverseLookupWithHint(linkage_upper,
            superpacket, copy_branch, tr_serial, set_missing, nullptr, nullptr);
        if( !foundpacket)
            return nullptr;
    }
    int ridx = wrapper->reverseIndex();
    if( !( *foundpacket)->size() || (ridx >= ( *foundpacket)->size()))
        return nullptr;
    if(copy_branch) {
        if(( *foundpacket)->subpackets()->m_serial != tr_serial) {
            *foundpacket = make_local_shared<Packet>( **foundpacket);
            ( *foundpacket)->subpackets().reset(new PacketList( *( *foundpacket)->subpackets()));
            ( *foundpacket)->m_missing = ( *foundpacket)->m_missing || set_missing;
            ( *foundpacket)->subpackets()->m_serial = tr_serial;
        }
    }
    local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(ridx));
    if( !p || (p->node().m_link != linkage)) {
        return nullptr;
    }
    if(upperpacket) {
        *upperpacket = *foundpacket;
        *index = ridx;
    }
    return &p;
}

// forwardLookup() — fallback tree scan when reverseLookupWithHint() fails.
// Searches the packet tree top-down by matching subnodes[i]->m_link == this.
// More expensive (O(N) in tree size) but always correct.
template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing,
    local_shared_ptr<Packet> *upperpacket, int *index) const {
    assert(superpacket);
    if( !superpacket->subpackets())
        return nullptr;
    if(copy_branch) {
        if(superpacket->subpackets()->m_serial != tr_serial) {
            superpacket = make_local_shared<Packet>( *superpacket);
            superpacket->subpackets().reset(new PacketList( *superpacket->subpackets()));
            superpacket->subpackets()->m_serial = tr_serial;
            superpacket->m_missing = superpacket->m_missing || set_missing;
        }
    }
    for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
        if(( *superpacket->subnodes())[i].get() == this) {
            local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
            if(subpacket) {
                *upperpacket = superpacket;
                *index = i;
                return &subpacket;
            }
        }
    }
    for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
        local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
        if(subpacket) {
            if(local_shared_ptr<Packet> *p =
                forwardLookup(subpacket, copy_branch, tr_serial, set_missing, upperpacket, index)) {
                return p;
            }
        }
    }
    return nullptr;
}

// reverseLookup() — find this node's packet within a transaction packet tree.
// Tries the fast hint-based path first; falls back to forward scan.
// When copy_branch is true, copy-on-write is applied along the found path.
template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode) {
    local_shared_ptr<Packet> *foundpacket;
    if( &superpacket->node() == this) {
        foundpacket = &superpacket;
    }
    else {
        local_shared_ptr<Packet> upperpacket;
        int index;
        foundpacket = reverseLookupWithHint(m_link, superpacket,
            copy_branch, tr_serial, set_missing, &upperpacket, &index);
        if(foundpacket) {
//				printf("$");
        }
        else {
//				printf("!");
            foundpacket = forwardLookup(superpacket, copy_branch, tr_serial, set_missing,
                &upperpacket, &index);
            if( !foundpacket)
                return 0;
        }
        if(uppernode)
            *uppernode = static_cast<XN*>(&upperpacket->node());
        assert( &( *foundpacket)->node() == this);
    }
    if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
        *foundpacket = make_local_shared<Packet>( **foundpacket);
    }
//						printf("#");
    return foundpacket;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing) {
    local_shared_ptr<Packet> *foundpacket = reverseLookup(superpacket, copy_branch, tr_serial, set_missing, 0);
    if( !foundpacket) {
        fprintf(stderr, "Node not found during a lookup.\n");
        throw NodeNotFoundError("Lookup failure.");
    }
    return *foundpacket;
}

template <class XN>
const local_shared_ptr<typename Node<XN>::Packet> &
Node<XN>::reverseLookup(const local_shared_ptr<Packet> &superpacket) const {
    local_shared_ptr<Packet> *foundpacket = const_cast<Node*>(this)->reverseLookup(
        const_cast<local_shared_ptr<Packet> &>(superpacket), false, 0, false, 0);
    if( !foundpacket) {
        fprintf(stderr, "Node not found during a lookup.\n");
        throw NodeNotFoundError("Lookup failure.");
    }
    return *foundpacket;
}

template <class XN>
XN *
Node<XN>::upperNode(Snapshot<XN> &shot) {
    XN *uppernode = 0;
    reverseLookup(shot.m_packet, false, 0, false, &uppernode);
    return uppernode;
}

//=============================================================================
//=============================================================================
// ascendOneLevel() — read bundledBy and prepare to go one level up
//
// Reads incoming_wrapper (const &) to extract bundledBy, reverseIndex.
// On success, loads the parent's wrapper into r.parent_wrapper via
// *r.parent_linkage (one load_shared_ ≈ 3 atomic ops).
//
// incoming_wrapper is NOT consumed — the caller keeps it alive for the
// staleness check at Step D (`*child_linkage != incoming_wrapper`).
// ABA safety: the incoming_wrapper's refcount is held by the caller's
// frame (Level 0: ScopedNegotiateLinkage view or view_copy() temporary;
// Level N: the previous level's r.parent_wrapper).
//
// No child_wrapper field is needed in WalkUpResult.
//=============================================================================
template <class XN>
inline typename Node<XN>::WalkUpResult
Node<XN>::ascendOneLevel(
    const local_shared_ptr<Linkage> &child_linkage,
    const ScopedNegotiateLinkage<XN> &incoming_scope) {

    WalkUpResult r;
    r.parent_packet = nullptr;
    assert( !incoming_scope->hasPriority());
    r.parent_linkage = incoming_scope->bundledBy();
    if( !r.parent_linkage) {
        r.find_status = (incoming_scope == *child_linkage)
            ? SnapshotStatus::NODE_MISSING : SnapshotStatus::DISTURBED;
        return r;
    }
    r.reverse_index = incoming_scope->reverseIndex();
    // Acquire parent via ScopedNeg (1 CAS, with_negotiate=false).
    // retry=0 + no contention → dtor is a plain view release (no tag, no wait).
    // On DISTURBED unwind the dtor tags contention on the parent linkage.
    // (Pass __LINE__ explicitly: emplace() resolves __builtin_LINE() inside
    //  <optional>, which would attribute these to a libc++ header line.)
    r.parent_scope.emplace(r.parent_linkage, incoming_scope.snap(), 0,
        ScopedNegotiateLinkage<XN>::TagMode::OnEntry, 2.0f, __LINE__);
    if( !*r.parent_scope) {
        r.find_status = SnapshotStatus::DISTURBED;
        return r;
    }
    r.find_status = SnapshotStatus::SUCCESS;
    return r;
}

//=============================================================================
// convertRecursiveStatus() — translate recursive call's status for this level
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::convertRecursiveStatus(
    SnapshotStatus recursive_status,
    WalkUpResult &r,
    local_shared_ptr<Packet> *&parent_packet) {

    switch(recursive_status) {
    case SnapshotStatus::DISTURBED:
    default:
        return recursive_status;
    case SnapshotStatus::VOID_PACKET:
    case SnapshotStatus::NODE_MISSING:
        // parent_packet points into r.parent_scope's PacketWrapper.
        // r.parent_scope (in the caller's WalkUpResult) keeps it alive.
        parent_packet = &(*r.parent_scope)->packet();
        r.is_root_level = true;
        return SnapshotStatus::SUCCESS;
    case SnapshotStatus::NODE_MISSING_AND_COLLIDED:
        parent_packet = &(*r.parent_scope)->packet();
        r.is_root_level = true;
        return SnapshotStatus::COLLIDED;
    case SnapshotStatus::COLLIDED:
    case SnapshotStatus::SUCCESS:
        r.is_root_level = false;
        return recursive_status;
    }
}

//=============================================================================
// findChildSlot() — scan parent's subnodes to find child's sub-packet slot
//
// Starting from reverse_index hint, wraps around to find the child whose
// m_link matches child_linkage. The hint may be stale if swap() reordered
// children.
//
// Returns:
//   current_status  if found (child_subpacket_out is set)
//   VOID_PACKET     if slot exists but sub-packet is null (missing child)
//   NODE_MISSING    if child not found in parent's subnodes
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::findChildSlot(
    const local_shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<Packet> *parent_packet,
    local_shared_ptr<Packet> **child_subpacket_out,
    int &reverse_index,
    SnapshotStatus current_status) {

    assert( *parent_packet);
    int size = ( *parent_packet)->size();
    int i = reverse_index;
    for(int cnt = 0;; ++cnt) {
        if(cnt >= size) {
            if(current_status == SnapshotStatus::COLLIDED)
                return SnapshotStatus::NODE_MISSING;
            return SnapshotStatus::NODE_MISSING;
        }
        if(i >= size)
            i = 0;
        if(( *( *parent_packet)->subnodes())[i]->m_link == child_linkage) {
            // Child node found at index i.
            *child_subpacket_out = &( *( *parent_packet)->subpackets())[i];
            reverse_index = i;
            if( !**child_subpacket_out) {
                assert(( *parent_packet)->missing());
                return SnapshotStatus::VOID_PACKET;
            }
            return current_status;  // SUCCESS or COLLIDED — child has a sub-packet.
        }
        ++i;
    }
}

//=============================================================================
// walkUpChainImpl() — common chain-walk: Steps A → B → C → D → E
//
// Shared by walkUpChain() and snapshotForUnbundle().
// Only the recursive call (Step B) differs between callers — passed as Recurser.
// Steps A–E are executed here; the result (with context) is returned via
// Steps:
//   A. ascendOneLevel: read bundledBy, save child_wrapper, return WalkUpResult
//   B. recurse via Recurser if parent is bundled
//   C. convertRecursiveStatus: map recursive result to this-level status
//   D. staleness check: verify child's linkage hasn't changed
//   E. findChildSlot: locate child's sub-packet in parent's packet
//=============================================================================
template <class XN>
template <class Recurser>
inline typename Node<XN>::WalkUpResult
Node<XN>::walkUpChainImpl(const local_shared_ptr<Linkage> &child_linkage,
    const ScopedNegotiateLinkage<XN> &incoming_scope,
    local_shared_ptr<Packet> **child_subpacket_out,
    Recurser &&recurse) {

    // Step A: ascend one level — reads incoming_scope (const &),
    // acquires parent into r.parent_scope (ScopedNeg, 1 CAS).
    WalkUpResult r = ascendOneLevel(child_linkage, incoming_scope);
    if(r.find_status != SnapshotStatus::SUCCESS)
        return r;

    // Step B: recurse if parent is also bundled.
    // Pass *r.parent_scope by const & — zero copy.
    // ascendOneLevel reads it without consuming it; parent_scope
    // remains intact for Step F.
    SnapshotStatus recursive_status = SnapshotStatus::NODE_MISSING;
    local_shared_ptr<Packet> *parent_packet;
    if( !(*r.parent_scope)->hasPriority()) {
        recursive_status = recurse(r.parent_linkage,
            *r.parent_scope, &parent_packet);
    }

    // Step C: convert recursive result — sets is_root_level.
    SnapshotStatus status = convertRecursiveStatus(
        recursive_status, r, parent_packet);
    if(status == SnapshotStatus::DISTURBED) {
        r.find_status = SnapshotStatus::DISTURBED;
        return r;
    }

    // Step D: staleness check — compare child_linkage against
    // incoming_scope (const & kept alive by the caller's frame).
    // No child_wrapper field needed; ABA prevented by caller's refcount.
    if(incoming_scope != *child_linkage) {
        r.find_status = SnapshotStatus::DISTURBED;
        return r;
    }

    // Step E: find child's sub-packet slot in parent's packet.
    r.find_status = findChildSlot(child_linkage, parent_packet,
        child_subpacket_out, r.reverse_index, status);
    r.status = status;
    r.parent_packet = parent_packet;
    return r;
}

//=============================================================================
// walkUpChain() — walk up the chain for snapshot/bundle (no CAS construction)
//   root_lifetime: receives the root-level PacketWrapper (via move) to keep
//   the Packet tree alive — foundpacket returned to the caller points into it.
//   snapshotForUnbundle doesn't need this because CASInfo keeps wrappers alive.
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::walkUpChain(const local_shared_ptr<Linkage> &child_linkage,
    const ScopedNegotiateLinkage<XN> &incoming_scope,
    local_shared_ptr<Packet> **child_subpacket_out,
    std::optional<ScopedNegotiateLinkage<XN>> &root_lifetime) {

    auto r = walkUpChainImpl(child_linkage, incoming_scope, child_subpacket_out,
        [&root_lifetime](const local_shared_ptr<Linkage> &pl,
           const ScopedNegotiateLinkage<XN> &is,
           local_shared_ptr<Packet> **pp) {
            return walkUpChain(pl, is, pp, root_lifetime);
        });
    if(r.is_root_level)
        root_lifetime = std::move(r.parent_scope);
    return r.find_status;
}

//=============================================================================
// snapshotForUnbundle() — walk up the chain with CAS info construction
//
// Calls walkUpChainImpl for Steps A–E, then performs Step F (CAS preparation)
// using the WalkUpResult context.
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::snapshotForUnbundle(const local_shared_ptr<Linkage> &child_linkage,
    const ScopedNegotiateLinkage<XN> &incoming_scope,
    local_shared_ptr<Packet> **child_subpacket_out,
    int64_t serial, CASInfoList *cas_infos) {

    auto r = walkUpChainImpl(child_linkage, incoming_scope, child_subpacket_out,
        [serial, cas_infos](const local_shared_ptr<Linkage> &pl,
           const ScopedNegotiateLinkage<XN> &is,
           local_shared_ptr<Packet> **pp) {
            return snapshotForUnbundle(pl, is, pp, serial, cas_infos);
        });

    // --- Post-processing for unbundle (Step F) ---
    if(r.find_status == SnapshotStatus::DISTURBED)
        return SnapshotStatus::DISTURBED;
    if(r.find_status == SnapshotStatus::VOID_PACKET) {
        cas_infos->clear();
        return SnapshotStatus::VOID_PACKET;
    }
    if(r.find_status == SnapshotStatus::NODE_MISSING && !r.parent_packet) {
        return SnapshotStatus::NODE_MISSING;
    }
    SnapshotStatus status = r.status;
    if(r.find_status == SnapshotStatus::NODE_MISSING) {
        if(status == SnapshotStatus::COLLIDED)
            return SnapshotStatus::NODE_MISSING;
        status = SnapshotStatus::NODE_MISSING;
    }

    // Identity check intentionally omitted:
    //   r.parent_wrapper->packet()->node().m_link == r.parent_linkage
    //   ( *r.parent_packet)->node().m_link == r.parent_linkage
    // Both can transiently fail when insert(online_after_insertion=true)
    // or release() publish the child's bundledBy and the parent's
    // linkage in two phases (the child's bundledBy must be published
    // first so reverseLookup works; the parent's linkage update lags).
    // The two are semantically atomic but cannot be made physically so.
    //
    // We do NOT short-circuit to DISTURBED here — that would force
    // retries in cases where the eventual CAS in unbundle() would have
    // succeeded (the transient resolves by CAS time), causing livelock.
    // Instead the cas_infos-driven CAS at unbundle() acts as the
    // natural race detector: if the observed wrapper has changed at
    // CAS time, the CAS fails and unbundle() returns DISTURBED, so the
    // caller retries.

    // CAS preparation
    if(status == SnapshotStatus::COLLIDED)
        return SnapshotStatus::COLLIDED;

    if((serial != SerialGenerator::SERIAL_NULL) &&
        ((*r.parent_scope)->m_bundle_serial == serial)) {
        if(status == SnapshotStatus::NODE_MISSING)
            return SnapshotStatus::NODE_MISSING;
        return SnapshotStatus::COLLIDED;
    }

    // Build new wrapper for this ancestor level.
    local_shared_ptr<Packet> *p(r.parent_packet);
    local_shared_ptr<PacketWrapper> newwrapper;
    if(r.is_root_level) {
        newwrapper =
            make_local_shared<PacketWrapper>( **r.parent_scope, (*r.parent_scope)->m_bundle_serial);
    }
    else {
        assert(cas_infos->size());
        // Use the root level's new_wrapper bundle_serial instead of
        // root_wrapper->m_bundle_serial.  They carry the same value
        // (root level pushes first with the root ancestor's serial,
        // and subsequent levels copy the same value).  This avoids
        // a post-recursion READ of root_wrapper, which is the
        // prerequisite for converting root_wrapper to a lighter type.
        newwrapper =
            make_local_shared<PacketWrapper>( *p, cas_infos->front().new_wrapper->m_bundle_serial);
    }
    if(newwrapper) {
        // Extract scoped_atomic_view from parent_scope into CASInfo.
        // parent_scope is not used after this point (parent_packet still
        // points into the PacketWrapper kept alive by the CASInfo's view).
        // The ScopedNeg's dtor handles its empty-view case cleanly.
        cas_infos->emplace_back(r.parent_linkage,
            r.parent_scope->consume_scoped_view(),
            newwrapper);
        p = &newwrapper->packet();
    }
    int size = ( *r.parent_packet)->size();
    if(size) {
        *p = make_local_shared<Packet>( **p);
        ( *p)->m_missing = true;
    }

    if((status == SnapshotStatus::NODE_MISSING) && (serial != SerialGenerator::SERIAL_NULL) &&
        (( !incoming_scope->hasPriority()) && (incoming_scope->m_bundle_serial == serial))) {
        printf("!");
        return SnapshotStatus::NODE_MISSING_AND_COLLIDED;
    }

    return status;
}

//=============================================================================
// snapshot() — obtain an immutable, consistent view of a subtree
//   (Comments by Claude Opus — based on source code analysis)
//
// Three cases depending on the node's current state:
//   1. hasPriority() and not missing: the node owns its packet directly —
//      just return the packet (fast path for leaf or already-bundled nodes).
//   2. !hasPriority(): the node is bundled into a super-node — walk up via
//      walkUpChain() to find the sub-packet, possibly triggering
//      unbundle() if the packet is incomplete (missing sub-packets).
//   3. hasPriority() but missing: sub-packets are stale — call bundle() to
//      absorb all child packets into a consistent parent packet.
//
// The Lamport serial (snapshot.m_serial) is generated from the node's
// m_bundle_serial so that the snapshot serial is always greater than the
// committed state it observes. After bundle(), gen() is called again to
// capture any Lamport advances that occurred during the recursive bundling.
//=============================================================================

// Out-of-line so that user-facing TUs which include only transaction.h
// (driver headers, modules, etc.) don't need the full ScopedNeg
// definition to instantiate Transaction<XN>::Transaction.  The outer
// ScopedNeg here is a "negotiate + load initial view" wrapper: after
// `commit()` we consume the view (zero atomic ops) and thread it into
// the 3-arg `snapshot` overload, sparing its first iteration the
// view-acquire load.
template <class XN>
void
Node<XN>::snapshot(Transaction<XN> &target, bool multi_nodal) const {
    scoped_atomic_view<PacketWrapper> initial_view;
    {
        ScopedNegotiateLinkage<XN> scope(m_link, target, -1,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry, 4.0f);
        scope.commit();
        initial_view = scope.consume_scoped_view();
    }
    snapshot(static_cast<Snapshot<XN> &>(target), multi_nodal,
             std::move(initial_view));
    target.m_oldpacket = target.m_packet;
}

template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal,
                   scoped_atomic_view<PacketWrapper> &&initial_view) const {
    auto &started_time = snapshot.m_started_time;
    auto &tid_bitset = snapshot.m_tid_bitset;
    struct GuardSnapshotRetryCount {
        GuardSnapshotRetryCount(Snapshot<XN> &s) : snapshot(s) {
            m_retry_count_started = snapshot.m_tx_retry_count;
        }
        ~GuardSnapshotRetryCount() {
            snapshot.m_tx_retry_count = m_retry_count_started;
        }
        Snapshot<XN> &snapshot;
        uint32_t m_retry_count_started;
    } guard(snapshot);

    for(int retry = 0;; ++retry) {
        if(retry)
            ++snapshot.m_tx_retry_count;
        // First iter: if caller supplied a pre-loaded view (e.g. from
        // the outer ScopedNeg in snapshot(Transaction&, ...) wrap),
        // use the move-in ctor (zero negotiate, zero view-acquire).
        // Subsequent iters: standard ctor (full negotiate + acquire).
        std::optional<ScopedNegotiateLinkage<XN>> scope_holder;
        if(retry == 0 && initial_view) {
            scope_holder.emplace(m_link, snapshot, retry,
                std::move(initial_view),
                ScopedNegotiateLinkage<XN>::TagMode::OnEntry,
                2.0f, /*with_negotiate=*/false);
        } else {
            scope_holder.emplace(m_link, snapshot, retry,
                ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        }
        ScopedNegotiateLinkage<XN> &scope = *scope_holder;
        if( !scope) {
            // Weak acquire CAS lost — treat as CAS failure: skip body
            // (m_contention_observed already set in ctor → dtor tags).
            continue;
        }
        snapshot.m_serial = SerialGenerator::gen(scope->m_bundle_serial);
        if(scope->hasPriority()) {
            if( !multi_nodal) {
                snapshot.m_packet = scope->packet();
                scope.commit();
                return;
            }
            if( !scope->packet()->missing()) {
                STRICT_assert(scope->packet()->checkConsistensy(scope->packet()));
                snapshot.m_packet = scope->packet();
                scope.commit();
                return;
            }
        }
        else {
            // Taking a snapshot inside the super packet.
            // Pass scope directly as incoming_scope (const ScopedNeg &)
            // — no view_copy() or temporary needed.
            // root_lifetime keeps the root-level ScopedNeg alive so that
            // foundpacket (pointing into the Packet tree) remains valid.
            local_shared_ptr<Linkage > linkage(m_link);
            local_shared_ptr<Packet> *foundpacket;
            std::optional<ScopedNegotiateLinkage<XN>> root_lifetime;
            auto status = walkUpChain(linkage,
                scope, &foundpacket, root_lifetime);
            switch(status) {
            case SnapshotStatus::SUCCESS: {
                    if( !( *foundpacket)->missing() || !multi_nodal) {
                        snapshot.m_packet = *foundpacket;
                        STRICT_assert(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
                        scope.commit();
                        return;
                    }
                    // The packet is imperfect, and then re-bundling the
                    // subpackets via unbundle.  Pass scope directly —
                    // its view IS the bundled_ref for the final CAS.
                    UnbundledStatus status = unbundle(nullptr, snapshot, scope);
                    switch(status) {
                    case UnbundledStatus::W_NEW_SUBVALUE:
                        // unbundle's final CAS via subscope succeeded —
                        // scope.m_committed already true.  Redundant
                        // commit() is a no-op but kept for clarity.
                        scope.commit();
                        break;
                    case UnbundledStatus::COLLIDED:
                    case UnbundledStatus::SUBVALUE_HAS_CHANGED:
                    default:
                        // unbundle() saw conflict from another thread.
                        scope.confirm_contention();
                        break;
                    }
                    continue;
                }
            case SnapshotStatus::DISTURBED:
            default:
                // walkUpChain disturbed by another thread; pre-CAS contention.
                scope.confirm_contention();
                continue;
            [[unlikely]] case SnapshotStatus::NODE_MISSING:
            [[unlikely]] case SnapshotStatus::VOID_PACKET:
                //The packet has been released.
                if( !scope->packet()->missing() || !multi_nodal) {
                    snapshot.m_packet = scope->packet();
                    scope.commit();
                    return;
                }
#if defined(KAME_STM_OPTIONAL_OPTIMIZATION) && KAME_STM_OPTIONAL_OPTIMIZATION
                // CAS-count optimization (not a correctness fix; see
                // VERIFICATION.md §5 and BundleUnbundle_hardlink_
                // nonatomic.tla).  Short-circuit the bundle-fall-
                // through's chain walk by self-promoting this node
                // directly: CAS a priority wrapper carrying the
                // current local packet.  On success the next loop iter
                // takes the hasPriority path; on CAS failure a peer
                // advanced m_link and the normal retry resumes.
                // Subtree absorption still happens via the subsequent
                // bundle() (priority+missing falls through).  The
                // ~10% test-time hang seen on
                // claude/refactor-negotiate-scoped-f7de2 (b23fa954)
                // is consistent with OS scheduling, not a logic gap.
                {
                    auto promoted = make_local_shared<PacketWrapper>(
                        scope->packet(), snapshot.m_serial);
                    scope.compareAndSet(promoted);
                }
                continue;
#else
                break;
#endif
            }
        }
        // Fall through to bundle.  Pass scope directly — bundle()
        // consumes scope's view at entry and restores via set_view on
        // success, so on SUCCESS we can read the post-bundle wrapper
        // through scope->packet().
        BundledStatus status = const_cast<Node *>(this)->bundle(
            scope, snapshot, snapshot.m_serial, true);
        switch (status) {
        case BundledStatus::SUCCESS:
            assert( !scope->packet()->missing());
            STRICT_assert(scope->packet()->checkConsistensy(scope->packet()));
            snapshot.m_serial = SerialGenerator::gen(); //Capture Lamport advances from bundle().
            snapshot.m_packet = scope->packet();
            scope.commit();
            return;
        default:
            // bundle() failed (DISTURBED); pre-CAS contention from
            // another thread's interference within bundle's own scopes.
            scope.confirm_contention();
            continue;
        }
    }
}

//=============================================================================
// bundle_subpacket() — prepare one child's packet for inclusion in a bundle
//   (Comments by Claude Opus — based on source code analysis)
//
// Handles the various states a child node can be in:
//   - Already bundled into this parent (linkage == m_link): if the sub-packet
//     is complete, nothing to do (SUCCESS). If missing, recursively bundle.
//   - Bundled into a *different* parent: unbundle it first (with collision
//     detection via bundle_serial), then recursively bundle if still missing.
//   - Has priority (owns its own packet): recursively bundle if missing.
//
// On success, subpacket_new is set to the child's complete packet.
// A null subpacket_new after SUCCESS means the child was already included
// in the current bundle (collision detected, deduplicated).
//=============================================================================
template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle_subpacket(ScopedNegotiateLinkage<XN> *supscope_super,
    const shared_ptr<Node> &subnode,
    ScopedNegotiateLinkage<XN> &subscope, local_shared_ptr<Packet> &subpacket_new,
    Snapshot<XN> &snap,
    int64_t bundle_serial) {
    auto &started_time = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    // Caller's subscope is on subnode->m_link.  We use its view directly
    // for unbundle / recursive bundle, eliminating the previous internal
    // subscope construction (which was a redundant move-in/move-out
    // dance through a temporary local_shared_ptr).

    if( !subscope->hasPriority()) {
        local_shared_ptr<Linkage > linkage(subscope->bundledBy());
        bool need_for_unbundle = false;
        bool detect_collision = false;
        if(linkage == m_link) {
            if(subpacket_new) {
                if(subpacket_new->missing()) {
                    need_for_unbundle = true;
                }
                else
                    return BundledStatus::SUCCESS;
            }
            else {
                if(subscope->packet()) {
                    //Re-inserted.
//					need_for_unbundle = true;
                }
                else
                    return BundledStatus::DISTURBED;
            }
        }
        else {
            need_for_unbundle = true;
            detect_collision = true;
        }
        if(need_for_unbundle) {
            // Pass subscope directly to unbundle.  unbundle's final CAS
            // (via subscope.compareAndSetWithHint) consumes subscope's
            // view on W_NEW_SUBVALUE; we restore it via set_view from
            // subwrapper_new so the rest of this function can continue
            // dereferencing subscope->.
            local_shared_ptr<PacketWrapper> subwrapper_new;
            UnbundledStatus status = unbundle(detect_collision ? &bundle_serial : nullptr, snap,
                subscope, nullptr, &subwrapper_new, supscope_super);
            switch(status) {
            case UnbundledStatus::W_NEW_SUBVALUE:
                // Final CAS in unbundle succeeded → subscope's view
                // is empty.  Move the new wrapper back into the view
                // (zero atomic ops, just pointer take).
                subscope.set_view(std::move(subwrapper_new));
                break;
            case UnbundledStatus::COLLIDED:
                // unbundle returned pre-CAS; subscope's view is still
                // valid.  Caller continues with the view as-is.
                subpacket_new.reset();
                return BundledStatus::SUCCESS;
            case UnbundledStatus::SUBVALUE_HAS_CHANGED:
            default:
                return BundledStatus::DISTURBED;
            }
        }
    }
    if(subscope->packet()->missing()) {
        assert(subscope->packet()->size());
        // Recursive bundle on subnode using the caller's subscope.
        // bundle() consumes the view at entry and restores via set_view
        // on SUCCESS (with the new bundled value).
        BundledStatus status = subnode->bundle(subscope, snap, bundle_serial, false);
        switch(status) {
        case BundledStatus::SUCCESS:
            break;
        case BundledStatus::DISTURBED:
        default:
            return BundledStatus::DISTURBED;
        }
    }
    subpacket_new = subscope->packet();
    return BundledStatus::SUCCESS;
}

//=============================================================================
// bundle() — multi-phase CAS protocol to absorb child packets into a parent
//   (Comments by Claude Opus — based on source code analysis)
//
// Purpose: make a subtree atomically snapshotable by collecting all child
//   node packets into the parent's packet. After bundling, a single
//   atomic_shared_ptr read on the parent's Linkage yields the entire
//   consistent subtree.
//
// Preconditions:
//   - oldsuperwrapper points to a valid PacketWrapper with a non-null,
//     missing packet (i.e. the packet knows it has stale child slots).
//   - bundle_serial is the Lamport serial for this bundling operation.
//
// Protocol phases (inside the retry loop):
//
//   Phase 1 — Collect sub-packets:
//     Create a copy of the parent packet. For each child node, read its
//     current PacketWrapper and call bundle_subpacket() to obtain the
//     child's packet (recursively bundling if the child itself is missing).
//     If a child is bundled into a *different* super-node, unbundle() it
//     first. Advances the Lamport clock past each child's serial.
//
//   Phase 2 — First checkpoint (CAS on parent Linkage):
//     Atomically replace the parent's PacketWrapper with the new one that
//     contains the collected sub-packets (still marked missing). If another
//     thread modified the parent, return DISTURBED and let the caller retry.
//
//   Phase 3 — Second checkpoint (CAS on each child Linkage):
//     Replace each child's PacketWrapper with a bundled_ref — a wrapper
//     that points back to the parent (bundledBy = m_link) and stores the
//     child's reverse index. These back-references allow the child to find
//     its packet inside the parent's bundled packet.
//     If any child was modified between phases 2 and 3, restart from phase 1.
//
//   Phase 4 — Finalize:
//     Create a final PacketWrapper. If no sub-packets are still missing,
//     clear the missing flag (the subtree is now fully consistent).
//     CAS the parent's Linkage one last time.
//
// The is_bundle_root flag (true only for the outermost bundle() call from
// snapshot()) forces missing=false at phase 4, because the root bundler
// will not be bundled further and needs a complete packet.
//=============================================================================
template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(ScopedNegotiateLinkage<XN> &supscope,
    Snapshot<XN> &snap,
    int64_t bundle_serial, bool is_bundle_root) {
    // Mark every linkage we tag during this bundle (via tag_as_contender)
    // with op_kind = BUNDLE.  Read side (peer-piggyback) not yet wired —
    // see VERIFICATION.md / paper notes.
    detail::ScopedOpKind _op_kind_scope(detail::StampKind::BUNDLE);
    auto &started_time = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    // Use supscope.view directly throughout — it tracks the current
    // m_link state across phases.  set_view updates after each
    // successful CAS; unbundle's cas_infos loop also updates via
    // set_view when an ancestor advances.  Saves the consume_view +
    // local var ledger management of the previous design.

    assert(supscope->packet());
    assert(supscope->packet()->size());
    assert(supscope->packet()->missing());

    Node &supernode(supscope->packet()->node());

    if( !supscope->hasPriority() ||
        (supscope->m_bundle_serial != bundle_serial)) {
        //Tags serial.
        // Keep local_shared_ptr (not unique_ptr) here: superwrapper is
        // moved into supscope.set_view after CAS for tracking.  unique_ptr
        // would lose the ref on CAS release, requiring a fetch_add to
        // re-acquire for supscope.  local_shared_ptr's CAS does fetch_add
        // internally (newr +1 for m_ref's implicit), and the caller's
        // local_shared_ptr's +1 transfers cleanly into supscope.view via
        // move-in set_view (0 ops).  Net: same atomic ops, simpler flow.
        local_shared_ptr<PacketWrapper> superwrapper(
            make_local_shared<PacketWrapper>(supscope->packet(), bundle_serial));
        ScopedNegotiateLinkage<XN> scope(supernode.m_link, snap, -1,
            ScopedNegotiateLinkage<XN>::TagMode::OnExit);
        if( !scope) return BundledStatus::DISTURBED;

#if defined(KAME_STM_OPTIONAL_OPTIMIZATION) && KAME_STM_OPTIONAL_OPTIMIZATION
        //Optional optimization:
        // Peer-completed early return: scope's fresh load of
        // supernode.m_link may show that a peer thread bundled this
        // subtree while we were negotiating.  Require hasPriority()
        // because a bundled-by-ancestor wrapper has m_packet=null
        // (m_bundledBy is the redirect target) — packet()->missing()
        // would crash there.  hasPriority() guarantees the wrapper
        // owns its packet, and !missing() means the bundle phase is
        // complete (Phase 4 cleared the flag).
        if(scope->hasPriority() && scope->packet()
           && !scope->packet()->missing()) {
            supscope = std::move(scope);
            return BundledStatus::SUCCESS;
        }
#endif
        // Pointer check + 1-arg CAS: scope loaded m_link at construction;
        // if it matches supscope's expected state, scope's internal view
        // serves as the CAS expected.  Saves 1 fetch_add + promote vs
        // view_copy().
        if(scope.operator->() != supscope.operator->()) {
            scope.confirm_contention();
            return BundledStatus::DISTURBED;
        }
        if( !scope.compareAndSet(superwrapper))
            return BundledStatus::DISTURBED;
        // CAS success: m_link advanced to superwrapper.  Update
        // supscope's view (move-in: 0 ops; supscope's old view is
        // released by set_view's internal release_).
        supscope.set_view(std::move(superwrapper));
    }

    fast_vector<scoped_atomic_view<PacketWrapper>, 16> subwrappers_org(supscope->packet()->subpackets()->size());

    for(int retry = 0;; ++retry) {
        // RAII OnEntry: negotiates supernode.m_link + tags eagerly (retry > 0).
        ScopedNegotiateLinkage<XN> scope(supernode.m_link, snap, retry,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        if( !scope) continue;  // weak acquire lost — retry
        // Peer-completed early return — same idea as the serial-tag
        // block above: skip Phases 1..4 entirely if peer has already
        // produced a non-missing, self-bundled wrapper at
        // supernode.m_link while we were negotiating.  hasPriority()
        // guard rejects bundled-by-ancestor wrappers (packet() == null).
        // if(scope->hasPriority() && scope->packet()
        //    && !scope->packet()->missing()) {
        //     supscope = std::move(scope);
        //     return BundledStatus::SUCCESS;
        // }
        local_shared_ptr<PacketWrapper> superwrapper(
            make_local_shared<PacketWrapper>( *supscope, bundle_serial));
        local_shared_ptr<Packet> &newpacket(
            reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
        assert(newpacket->size());
        assert(newpacket->missing());

        STRICT_assert(s_serial_abandoned != newpacket->subpackets()->m_serial);

        //--- Phase 1: collect sub-packets from child nodes ---
        newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));
        shared_ptr<PacketList> &subpackets(newpacket->subpackets());
        shared_ptr<NodeList> &subnodes(newpacket->subnodes());

        bool missing = false;
        for(unsigned int i = 0; i < subpackets->size(); ++i) {
            shared_ptr<Node> child(( *subnodes)[i]);
            local_shared_ptr<Packet> &subpacket_new(( *subpackets)[i]);
            for(int child_retry = 0;; ++child_retry) {
                // RAII OnEntry: negotiates child->m_link + tags eagerly
                // on retry > 0.
                ScopedNegotiateLinkage<XN> child_scope(
                    child->m_link, snap, child_retry,
                    ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
                if( !child_scope) continue;  // weak acquire lost — retry
                // Fast-path: child's m_link unchanged since last iter's
                // observation.  Compare child_scope's view directly
                // (operator== on ref pointer, no atomic op).
                if(child_scope == subwrappers_org[i]) {
                    child_scope.commit(); //fast path for retry > 0.
                    break;
                }
                SerialGenerator::gen(child_scope->m_bundle_serial); //Lamport: advance past sub-node serial.
                // Pass child_scope directly — bundle_subpacket uses its
                // view for unbundle / recursive bundle, eliminating the
                // previous view_copy + temporary subscope dance.
                BundledStatus status = bundle_subpacket( &supscope,
                    child, child_scope, subpacket_new, snap, bundle_serial);
                switch(status) {
                case BundledStatus::SUCCESS:
                    break;
                case BundledStatus::DISTURBED:
                default:
                    child_scope.confirm_contention();
                    if(supscope == *supernode.m_link)
                        continue;
                    scope.confirm_contention();
                    return status;
                }
                // Move child_scope's view into subwrappers_org[i].
                // ZERO atomic ops — direct scoped_atomic_view move.
                // child_scope's view becomes empty; dtor is a no-op
                // (child_scope.commit() below prevents tagging).
                subwrappers_org[i] = child_scope.consume_scoped_view();
                if(subpacket_new) {
                    if(subpacket_new->missing()) {
                        missing = true;
                    }
                    assert(&subpacket_new->node() == child.get());
                }
                else
                    missing = true;
                child_scope.commit();
                break;
            }
        }
        if(is_bundle_root) {
            assert( &supernode == this);
            missing = false;
        }
        newpacket->m_missing = true;

        //--- Phase 2: first checkpoint — CAS the parent PacketWrapper ---
        // Pointer check + 1-arg CAS: scope's view was loaded from
        // supernode.m_link at scope creation; if it matches supscope's
        // expected state, the internal view serves as CAS expected.
        // compareAndSetRetain: on success scope transitions to
        // Owned(superwrapper) — scope's view tracks the new m_link
        // value through Phase 3, ready for Phase 4's CAS without reload.
        if(scope.operator->() != supscope.operator->()) {
            scope.confirm_contention();
            return BundledStatus::DISTURBED;
        }
        if( !scope.compareAndSetRetain(superwrapper))
            return BundledStatus::DISTURBED;
        // Update supscope.view to track the new m_link state.
        // Pass copy of superwrapper (still needed for Phase 4).
        supscope.set_view(local_shared_ptr<PacketWrapper>(superwrapper));

        //--- Phase 3: second checkpoint — CAS each child's Linkage to point back to parent ---
        //  Each bundled_ref is a PacketWrapper holding a back-reference
        //  (bundledBy → parent's m_link) and the child's reverse index.
        bool changed_during_bundling = false;
        for(unsigned int i = 0; i < subnodes->size(); i++) {
            shared_ptr<Node> child(( *subnodes)[i]);
            local_shared_ptr<PacketWrapper> bundled_ref;
            if(( *subpackets)[i])
                bundled_ref = make_local_shared<PacketWrapper>(m_link, i, bundle_serial);
            else
                bundled_ref = make_local_shared<PacketWrapper>( *subwrappers_org[i], bundle_serial);

            assert( !bundled_ref->hasPriority());
            //Second checkpoint, the written bundle is valid or not.
            ScopedNegotiateLinkage<XN> childScope(child->m_link, snap, retry,
                ScopedNegotiateLinkage<XN>::TagMode::OnExit,
                2.0f / subnodes->size());
            if( !childScope) {
                // Weak acquire lost — treat as Phase 3 CAS failure.
                // No need to manually release subwrappers_org[i..n-1]:
                // ADAPTIVE_THRESHOLD ensures non-privileged threads
                // drain their tags on saturation (rcnt >= LOCAL_CAP-2),
                // so peer-thread acquires will eventually drain our
                // parked TagHelds via promote.
                changed_during_bundling = true;
                break;
            }
            // Pointer check + 1-arg CAS: verify childScope loaded
            // the same wrapper we saved in Phase 1, then CAS to
            // bundled_ref.  Saves view_copy's fetch_add + promote.
            {
                bool child_cas_ok =
                    (childScope.operator->() == subwrappers_org[i].get());
                if(child_cas_ok)
                    child_cas_ok = childScope.compareAndSet(bundled_ref);
                if(child_cas_ok) {
                    // Phase 3 child-CAS succeeded: this child's linkage
                    // just transitioned from "owns its own packet" to
                    // BundledRefWrapper (bundled into supernode).
                    // From the child linkage's perspective this is a
                    // real BUNDLE event — record it so the per-Linkage
                    // flip detector sees the BUNDLE side too, not only
                    // the supernode's Phase 4 record.  Symmetric with
                    // unbundle's cas_infos loop UNBUNDLE record.
                    // Deferred to childScope's dtor so the publish
                    // fires after the chain has settled past this
                    // sub-CAS — peers reading m_recent_ops_state see
                    // events more strongly correlated with "CAS truly
                    // done" rather than "CAS just happened".
#if KAME_ENABLE_SPIN_BAND_GATE
                    childScope.arm_record_on_commit(
                        NegotiationCounter::with_kind(started_time,
                                                      detail::StampKind::BUNDLE));
#endif
                }
                if( !child_cas_ok) {
                    // Phase 3 child-CAS failure (pointer mismatch or
                    // CAS lost).  confirm_contention is idempotent with
                    // compareAndSet's internal m_contention_observed.
                    childScope.confirm_contention();
                    {
                        scoped_atomic_view<PacketWrapper> child_check(
                            *child->m_link,
                            scoped_atomic_view<PacketWrapper>::ADAPTIVE_THRESHOLD,
                            /*weakly=*/true);
                        if(!child_check.acquire_succeeded()
                         || (child_check->m_bundle_serial != bundle_serial)
                         || (supscope != *supernode.m_link)) {
                            // Phase 2 CAS on supernode.m_link already
                            // succeeded; commit the outer scope before
                            // returning DISTURBED so its dtor doesn't
                            // re-tag/assert on legitimate forward progress.
                            scope.commit();
                            return BundledStatus::DISTURBED;
                        }
                    }
                    // No need to manually release subwrappers_org[i..n-1]:
                    // ADAPTIVE_THRESHOLD ensures peer-thread acquires
                    // promote-and-drain on saturation, naturally
                    // recovering tag space without explicit cleanup.
                    changed_during_bundling = true;
                    break;
                }
            }
            // childScope auto-committed via scope.compareAndSet.
        }
        if(changed_during_bundling) {
            // Phase 2 CAS already succeeded — supernode state advanced.
            // Mark the outer scope committed before re-iterating to
            // suppress the dtor tag/assert (continuation handles Phase 3
            // contention by retrying the entire bundle).
            scope.commit();
            continue;
        }

        //--- Phase 4: finalize — clear missing flag if all sub-packets are present ---
        superwrapper = make_local_shared<PacketWrapper>( *superwrapper, bundle_serial);
        if( !missing) {
            local_shared_ptr<Packet> &newpacket(
                reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
            // Hard-link race gate: verify every Null sub-slot is
            // reverseLookup-able before publishing ~missing (the
            // production "30/30 abort" pattern; see VERIFICATION.md §5
            // and tests/tlaplus/BundleUnbundle_hardlink_4node).  On
            // failure, restore m_missing=true and DISTURBED so the
            // outer retry re-attempts once the race resolves.
            //
            // Clear m_missing BEFORE the gate: both helpers short-
            // circuit their Null-slot reverseLookup when the observed
            // root is missing (mid-bundle).  Default globalroot is
            // correct: for `is_bundle_root=true` the reverseLookup
            // at line ~1440 makes newpacket alias superwrapper's
            // packet, i.e. the bundle's global root.
            newpacket->m_missing = false;
            if(newpacket->allSubReachable(newpacket)) [[likely]] {
                STRICT_assert(newpacket->checkConsistensy(newpacket));
            }
            else {
                newpacket->m_missing = true;
                scope.confirm_contention();
                scope.commit();
                return BundledStatus::DISTURBED;
            }
        }

        // CAS via scope.  Phase 2's compareAndSetRetain left scope in
        // Owned(Phase 2 newr) — no reload needed.  Pointer check
        // confirms no concurrent change to supernode.m_link since Phase 2.
        if(scope.operator->() != supscope.operator->()) {
            scope.confirm_contention();
            scope.commit();
            return BundledStatus::DISTURBED;
        }
        if( !scope.compareAndSetWithHint(superwrapper, started_time)) {
            scope.commit();
            return BundledStatus::DISTURBED;
        }
        // CAS success: m_link advanced.  If the new wrapper has its
        // missing flag cleared (Phase 4 finalize executed because all
        // children were collected), this is a *real* BUNDLE publish
        // event on supernode.m_link — record it on the recent-ops log.
        // Partial-bundle publishes (missing=true) do not count: peers
        // can't ride them as a coalesce result, so they shouldn't bias
        // the period EMA.
#if KAME_ENABLE_SPIN_BAND_GATE
        if( !missing) {
            supernode.m_link->template record_successful_op<NegotiationCounter>(
                NegotiationCounter::with_kind(started_time,
                                              detail::StampKind::BUNDLE));
        }
#endif
        // Update supscope.view to track.
        supscope.set_view(std::move(superwrapper));

        for(unsigned int i = 0; i < subnodes->size(); i++) {
            shared_ptr<Node> child(( *subnodes)[i]);
            //this tagging significantly increased a commiting rate.
            child->m_link->tags_successful_cas(started_time);
        }
        // scope.compareAndSetWithHint above already auto-committed +
        // tagged successful_cas on supernode.m_link.
        break;
    }
    return BundledStatus::SUCCESS;
}

//template <class XN>
//void
//Node<XN>::fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> > &subwrappers,
//	const local_shared_ptr<Packet> &packet) {
//	for(int i = 0; i < packet->size(); ++i) {
//		const local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
//		subwrappers.push_back( *( *packet->subnodes())[i]->m_link);
//		if(subpacket)
//			fetchSubpackets(subwrappers, subpacket);
//	}
//}
//template <class XN>
//bool
//Node<XN>::commit_at_super(Transaction<XN> &tr) {
//	Node &node(tr.m_packet->node());
//	for(Transaction<XN> tr_super( *this);; ++tr_super) {
//		local_shared_ptr<Packet> *packet
//			= node.reverseLookup(tr_super.m_packet, false, Packet::SERIAL_NULL, false, 0);
//		if( !packet)
//			return false; //Released.
//		if( *packet != tr.m_oldpacket) {
//			if( !tr.isMultiNodal() && (( *packet)->payload() == tr.m_oldpacket->payload())) {
//				//Single-node mode, the payload in the snapshot is unchanged.
//				tr.m_packet->subpackets() = ( *packet)->subpackets();
//				tr.m_packet->m_missing = ( *packet)->missing();
//			}
//			else {
//				return false;
//			}
//		}
//		node.reverseLookup(tr_super.m_packet, true, tr_super.m_serial, tr.m_packet->missing())
//			= tr.m_packet;
//		if(tr_super.commit()) {
//			tr.m_packet = tr_super.m_packet;
//			return true;
//		}
//	}
//}
//=============================================================================
// commit() — atomically publish a transaction's modified packet
//   (Comments by Claude Opus — based on source code analysis)
//
// Two paths depending on the node's current state:
//
//   hasPriority() = true (the node owns its Linkage directly):
//     Fast path — single CAS on m_link to replace the old PacketWrapper
//     with one containing the transaction's new packet.
//     Single-node optimization: if only the payload changed (not children),
//     and another transaction changed the children since our snapshot,
//     we can adopt the new children and still commit — avoiding a spurious
//     conflict when two transactions touch disjoint parts of the same node.
//
//   hasPriority() = false (the node is bundled into a super-node):
//     Must unbundle first to reclaim the Linkage, then CAS the new packet.
//     For multi-nodal transactions, the new packet is passed into unbundle
//     so it can be installed in a single step. For single-node transactions,
//     unbundle restores priority and commit retries via the fast path.
//=============================================================================
template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr) {
    assert(tr.m_oldpacket != tr.m_packet);
    assert(tr.isMultiNodal() || tr.m_packet->subpackets() == tr.m_oldpacket->subpackets());
    assert(this == &tr.m_packet->node());

    // Stamp every linkage tagged during commit with kind = BUNDLE,
    // but ONLY for multilevel (multi-nodal) transactions whose
    // commit involves nested bundle/unbundle on
    // tree linkages.  A SingleTransaction is a stand-alone single-
    // node CAS — structurally closer to a snapshot/release (no
    // enclosing retry loop with sub-operations), so it tags with
    // kind = NONE to share the normal adaptive-sleep fairness path
    // with those stand-alone ops.
    //
    // Downstream effect: the bundle/unbundle-level kind-gated gate-
    // return in negotiate_internal piggybacks only when peer is also
    // a multilevel Tx (kind ∈ {BUNDLE, UNBUNDLE}), preserving the
    // SingleTransaction's adaptive-sleep fairness window.  Multinodal
    // commits previously stamped `MultiNodalCommit` (=3) — now folded
    // into BUNDLE (functionally identical in every production path);
    // slot 3 is reserved for a per-Linkage privilege flag.
    detail::ScopedOpKind _op_kind_scope(
        tr.isMultiNodal() ? detail::StampKind::BUNDLE
                          : detail::StampKind::NONE);

    local_shared_ptr<PacketWrapper> newwrapper(make_local_shared<PacketWrapper>(tr.m_packet, tr.m_serial));
    for(int retry = 0;; ++retry) {
        // RAII OnEntry: negotiates + tag-bit acquires view of m_link
        // + tags eagerly (retry > 0).  scope's internal view is the
        // CAS oldr.
        ScopedNegotiateLinkage<XN> scope(m_link, tr, retry,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        if( !scope) {
            // Weak acquire CAS lost.  m_contention_observed already
            // set in ctor — dtor tags this iteration.  Skip the body.
            continue;
        }
        if(scope->hasPriority()) {
            //Committing directly to the node.
            if(scope->packet() != tr.m_oldpacket) {
                if( !tr.isMultiNodal() && (scope->packet()->payload() == tr.m_oldpacket->payload())) {
                    //Single-node mode, the payload in the snapshot is unchanged.
                    tr.m_packet->subpackets() = scope->packet()->subpackets();
                    tr.m_packet->m_missing = scope->packet()->missing();
                }
                else {
                    STRICT_TEST(s_serial_abandoned = tr.m_serial);
//					fprintf(stderr, "F");
                    // Conflict detected: another thread committed a new packet
                    // before we even reached the CAS.  Pre-CAS exit; no CAS
                    // attempted via the scope.  confirm_contention() forces
                    // dtor tag so retry==0 fast-path doesn't skip the
                    // contender mark.
                    scope.confirm_contention();
                    return false;
                }
            }
//			STRICT_TEST(std::deque<local_shared_ptr<PacketWrapper> > subwrappers);
//			STRICT_TEST(fetchSubpackets(subwrappers, wrapper->packet()));
            STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));

            // CAS via the scope: success → auto-commit + priority hint;
            // failure → m_contention_observed → dtor tag.
            if(scope.compareAndSetWithHint(newwrapper, tr.m_started_time))
                return true;
            continue;
        }
        //Unbundling this node from the super packet.
        // Pass the outer scope directly — its view IS the bundled_ref
        // (current value of m_link == sublinkage).  unbundle uses
        // subscope.compareAndSetWithHint() for the final CAS, no
        // separate inner negotiate.
        UnbundledStatus status = unbundle(nullptr, tr, scope,
            tr.isMultiNodal() ? &tr.m_oldpacket : nullptr, tr.isMultiNodal() ? &newwrapper : nullptr);
        switch(status) {
        case UnbundledStatus::W_NEW_SUBVALUE:
            if(tr.isMultiNodal()) {
                scope.commit();
                return true;
            }
            // unbundle() succeeded in its inner sublinkage CAS on m_link.
            // The outer scope here covers the same m_link; mark it
            // committed so the dtor neither tags (already tagged by the
            // inner CAS path) nor asserts.
            scope.commit();
            continue;  // RAII already tagged at ctor for retry > 0
        case UnbundledStatus::SUBVALUE_HAS_CHANGED: {
                STRICT_TEST(s_serial_abandoned = tr.m_serial);
//				fprintf(stderr, "F");
                // Pre-CAS conflict from another thread.
                scope.confirm_contention();
                return false;
            }
        case UnbundledStatus::DISTURBED:
        default:
            // unbundle() failed before touching m_link (ancestor CAS race,
            // or snapshotForUnbundle disturbed).  Pre-CAS contention.
            scope.confirm_contention();
            continue;
        }
    }
}

//=============================================================================
// unbundle() — extract a sub-node's packet from its super-node's bundle
//   (Comments by Claude Opus — based on source code analysis)
//
// Purpose: the reverse of bundle(). Restores a child node's Linkage to
//   point directly to its own PacketWrapper (hasPriority = true) so that
//   the child can be committed independently.
//
// Parameters:
//   - bundle_serial: if non-null, check for collisions (the sub-packet
//     may already be included in an ongoing snapshot with this serial).
//   - subscope: scope around the child's Linkage; its internal view is
//     the back-reference wrapper (formerly the `bundled_ref`
//     parameter).  The final sublinkage CAS uses subscope's view as
//     oldr — no re-negotiate before the final CAS (caller's outer
//     negotiate covers it).
//   - oldsubpacket: if provided, verify the sub-packet hasn't changed
//     (conflict detection for commit).
//   - newsubwrapper_returned: if provided, the new PacketWrapper for the
//     child is stored here (used by commit to install in one step).
//   - oldsuperwrapper: if provided, track super-node wrapper updates
//     (needed when commit must know the super-node's current state).
//
// Steps:
//   1. Walk up via snapshotForUnbundle() to locate the sub-packet inside
//      the super-node's bundled packet. snapshotForUnbundle() builds a
//      cas_infos list of (linkage, old_wrapper, new_wrapper) for each
//      ancestor that needs its PacketWrapper updated.  Walks on a
//      separate `superwrapper` value-typed wrapper because chain-walk
//      crosses atomic_shared_ptr boundaries (scoped_atomic_view binds
//      to one atomic).
//   2. If oldsubpacket was given and doesn't match, return
//      SUBVALUE_HAS_CHANGED (the transaction must fail).
//   3. Execute the CAS list from cas_infos bottom-up: each ancestor's
//      Linkage gets a new PacketWrapper with a copied packet that marks
//      the child's slot as missing (the child is leaving the bundle).
//   4. CAS the child's Linkage via subscope.compareAndSetWithHint:
//      replace the back-reference (subscope's view) with a new
//      PacketWrapper that holds the extracted sub-packet directly
//      (hasPriority = true). The serial is advanced past the super-
//      node's bundle serial via gen().
//=============================================================================
template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, Snapshot<XN> &snap,
    ScopedNegotiateLinkage<XN> &subscope,
    const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper_returned,
    ScopedNegotiateLinkage<XN> *supscope_super) {
    // Mark every linkage we tag during this unbundle (via tag_as_contender)
    // with op_kind = UNBUNDLE.  Read side not yet wired.
    detail::ScopedOpKind _op_kind_scope(detail::StampKind::UNBUNDLE);
    auto &time_started = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    assert( !subscope->hasPriority());

// Pass subscope directly as incoming_scope (const ScopedNeg &)
// — no view_copy() or temporary needed.
    local_shared_ptr<Packet> *newsubpacket;
    CASInfoList cas_infos;
    SnapshotStatus status = snapshotForUnbundle(subscope.linkage(),
        subscope, &newsubpacket,
        bundle_serial ? *bundle_serial : SerialGenerator::SERIAL_NULL, &cas_infos);
    if(status == SnapshotStatus::DISTURBED)
        return UnbundledStatus::DISTURBED;
    if(status == SnapshotStatus::VOID_PACKET || status == SnapshotStatus::NODE_MISSING) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &subscope->packet());
        assert(newsubpacket);
    }
    if(status == SnapshotStatus::NODE_MISSING_AND_COLLIDED) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &subscope->packet());
        assert(newsubpacket);
        status = SnapshotStatus::COLLIDED;
    }
    // SUCCESS, COLLIDED → fall through

    if(oldsubpacket && ( *newsubpacket != *oldsubpacket))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;

    // Save root's bundle_serial BEFORE the CAS loop — the loop may
    // move cas_infos entries' new_wrapper via set_view().
    // When cas_infos is non-empty, the root ancestor's bundle_serial
    // is in cas_infos.front() (root level pushes first).
    // When empty (NODE_MISSING — node not actually bundled), the
    // sub-node's own bundle_serial (via subscope) is the root serial.
    int64_t root_bundle_serial = cas_infos.empty()
        ? subscope->m_bundle_serial
        : cas_infos.front().new_wrapper->m_bundle_serial;

    for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
        // Save old wrapper identity before moving into scope.
        PacketWrapper *old_pw = it->old_wrapper.get();
        // Move CASInfo's scoped_atomic_view into ScopedNeg — reuses the
        // view directly (0 tag-acquire ops vs standard ctor's ~1 CAS).
        // with_negotiate=true: each cas_info is a different ancestor
        // linkage, so a fresh negotiate is needed.
        ScopedNegotiateLinkage<XN> scope(it->linkage, snap, -1,
            std::move(it->old_wrapper),
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry, 2.0f / cas_infos.size(),
            /*with_negotiate=*/true);
        if( !scope) return UnbundledStatus::DISTURBED;  // view was empty
        if( !scope.compareAndSet(it->new_wrapper))
            return UnbundledStatus::DISTURBED;
        // scope.compareAndSet auto-committed on success.  A subsequent
        // return DISTURBED below from the oldsuperwrapper check is
        // legitimate forward progress (linkage already advanced), and
        // m_committed=true silences the dtor's tag/assert.
        //
        // Each ancestor's packet just got replaced (subtree shrunk —
        // a descendant was extracted by this unbundle), so this is a
        // real UNBUNDLE event from THIS linkage's point of view too,
        // not only the final subscope.  Record it so the per-Linkage
        // flip detector can build up evidence on intermediate nodes
        // (which otherwise see BUNDLE only and never cross the LOW
        // band).  Deferred to scope dtor so the publish fires after
        // the cas_infos loop progresses past this ancestor (peer
        // reading m_recent_ops_state sees an event more strongly
        // correlated with "ancestor settled").
#if KAME_ENABLE_SPIN_BAND_GATE
        scope.arm_record_on_commit(
            NegotiationCounter::with_kind(time_started,
                                          detail::StampKind::UNBUNDLE));
#endif
        if(supscope_super) {
            if( ( *supscope_super)->packet()->node().m_link == it->linkage) {
                if(supscope_super->operator->() != old_pw)
                    return UnbundledStatus::DISTURBED;
                // Update super-scope's view to track ancestor's new
                // wrapper.  set_view (move version): release old view
                // (~1 op) + transfer new (0 ops).  it->new_wrapper is
                // local to cas_infos and not used after this iteration.
                supscope_super->set_view(std::move(it->new_wrapper));
            }
        }
    }
    if(status == SnapshotStatus::COLLIDED)
        return UnbundledStatus::COLLIDED;

    local_shared_ptr<PacketWrapper> newsubwrapper;
    if(oldsubpacket)
        newsubwrapper = *newsubwrapper_returned;
    else
        newsubwrapper = make_local_shared<PacketWrapper>( *newsubpacket, SerialGenerator::gen(root_bundle_serial));

    // Final sublinkage CAS via the caller-provided subscope.  The
    // outer scope's negotiate (at construction) covers this CAS — no
    // separate inner re-negotiate, saves one negotiate() per unbundle.
    if( !subscope.compareAndSetWithHint(newsubwrapper, time_started))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;
    // subscope.compareAndSetWithHint auto-committed + tagged success.
    // The new wrapper at subscope.linkage is hasPriority (its own
    // packet) — this is a real UNBUNDLE publish event.
#if KAME_ENABLE_SPIN_BAND_GATE
    subscope.linkage()->template record_successful_op<NegotiationCounter>(
        NegotiationCounter::with_kind(time_started,
                                      detail::StampKind::UNBUNDLE));
#endif

    for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
        it->linkage->tags_successful_cas(time_started);
    }

    if(newsubwrapper_returned)
        *newsubwrapper_returned = newsubwrapper;

//	if(oldsuperwrapper) {
//		if( &( *oldsuperwrapper)->packet()->node().m_link != &cas_infos.front().linkage)
//			return UNBUNDLE_DISTURBED;
//		if( *oldsuperwrapper != cas_infos.front().old_wrapper)
//			return UNBUNDLE_DISTURBED;
//		printf("1\n");
//		*oldsuperwrapper = cas_infos.front().new_wrapper;
//	}

    return UnbundledStatus::W_NEW_SUBVALUE;
}

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #include <windows.h>
#endif

void setCurrentPriorityMode(Priority pr) {
    *stl_currentPriority = pr;
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    SetThreadPriority(GetCurrentThread(),
        (pr == Priority::HIGHEST) ? THREAD_PRIORITY_TIME_CRITICAL : THREAD_PRIORITY_NORMAL);
#endif
}

} //namespace Transactional

