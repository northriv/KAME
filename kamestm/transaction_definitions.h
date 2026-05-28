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
#ifndef TRANSACTION_DEFINITIONS_H
#define TRANSACTION_DEFINITIONS_H

#include <atomic>   // ATOMIC_LLONG_LOCK_FREE for KAME_STM_COMPACT_STATE

// --- 32-bit platform fallback: compact STM state ---------------------
//
// The STM negotiation/priority machinery packs several fields into
// `atomic<uint64_t>` and a 64-bit `cnt_t` stamp.  On 64-bit hosts these
// are trivially lock-free; on 32-bit hosts with DCAS (CMPXCHG8B on
// i486+, LDREXD/STREXD on ARMv7-A) the compiler emits a lock-free
// 64-bit atomic via std::atomic, so we keep the full layout there too.
//
// When the toolchain reports `ATOMIC_LLONG_LOCK_FREE != 2`
// (32-bit targets without hardware DCAS — e.g. i386 or ARMv5/v6, or a
// toolchain configured to fall back to a mutex), KAME_STM_COMPACT_STATE
// is auto-detected as 1.  In compact mode:
//   * `NegotiationCounter::cnt_t` is `int32_t`, stamp is [us:24|tid:8],
//     with the `lowprio` and `kind` fields sealed (no priv stamps, no
//     lowprio expiration — these are heuristic-only paths, not
//     correctness-critical).
//   * `Linkage::m_priority_state` is `atomic<uint32_t>`,
//     [tid:8|lease_us:8|start_us:16].
//   * `Linkage::m_recent_ops_state` is `atomic<uint32_t>` (unused;
//     forced off via KAME_ENABLE_SPIN_BAND_GATE=0).
//   * `RunnerCounterEntry::v` is `atomic<uint32_t>` (0/1 in steady state).
//   * `KAME_ENABLE_RUNNER_DIGEST` and `KAME_ENABLE_SPIN_BAND_GATE` are
//     forced to 0 (their diagnostic atomic<uint64_t> stores would
//     otherwise still require DCAS).
//
// The 8-bit TID range (256 values) collides for ProcessCounter::id()
// values > 255 — `tag_as_contender` / `i_am_privileged_now` /
// `fair_mode_blocks_me` may misidentify a different thread as "me",
// but these are heuristic fast paths: the lower CAS layer
// (`PacketWrapper` / Packet) is independent of TID, so data
// integrity is preserved.  Worst case is extra CAS retries.
//
// To override (e.g. force compact on a 64-bit host for testing):
//   -DKAME_STM_COMPACT_STATE=1 or -DKAME_STM_COMPACT_STATE=0
#ifndef KAME_STM_COMPACT_STATE
#  if ATOMIC_LLONG_LOCK_FREE == 2
#    define KAME_STM_COMPACT_STATE 0
#  else
#    define KAME_STM_COMPACT_STATE 1
#  endif
#endif

// =====================================================================
// Compile-time tuning knobs for the STM negotiation / livelock-free /
// adaptive-backoff / per-Linkage-flip / spin-for-same-kind machinery.
// Every macro here is `-D`-overridable at cmake time (or via the
// toolchain CXXFLAGS). Header is included from the top of
// transaction.h so the declarations of Node<XN>::Linkage /
// NegotiationCounter / ScopedNegotiateLinkage / NegSite can reference
// them.
//
// Defaults reflect the most recent sweep winners on iMac Pro / Apple
// Silicon / Linux x86; see git history for rationale per knob.
// =====================================================================

// Umbrella flag for opt-in STM optimizations that bypass the strictly
// TLA+-modelled code paths.  Default = 1 (enabled).
//
// Currently gates:
//   * `snapshot()` NODE_MISSING/VOID_PACKET self-promote CAS — a
//     one-CAS shortcut around the bundle-fall-through chain walk
//     when the local packet is "missing" (limbo state left by
//     `release()`).
//   * `bundle()` peer-completed early-return — fast path that
//     accepts a peer's already-bundled root and returns SUCCESS
//     without redoing Phase 1-4.
//
// NOTE: These shortcuts are NOT covered by the TLA+ suite under
// `tests/tlaplus/`.  The hard-link models (`_hardlink_*`) verify
// the unconditional paths only — they do not encode the
// short-circuit branches.  The non-atomic model
// (`BundleUnbundle_hardlink_nonatomic.tla`) does compare a
// "self-promote" finalize variant against the bundle-fall-through
// one, but both are shown live at the same modelling abstraction;
// the C++ shortcut is a CAS-count optimization rather than a
// liveness-required mechanism.
//
// Set `-DKAME_STM_OPTIONAL_OPTIMIZATION=0` to fall back to the
// strictly TLA+-modelled paths if a regression is suspected.
#ifndef KAME_STM_OPTIONAL_OPTIMIZATION
#define KAME_STM_OPTIONAL_OPTIMIZATION 1
#endif

// --- Per-Linkage priority / lease ------------------------------------

// Initial per-Linkage lease (ns). Stored as µs in the packed priority
// state (uint16_t field), so the runtime default of 10000 ns = 10 µs.
#ifndef KAME_LEASE_NS_BASE
#define KAME_LEASE_NS_BASE 10000    // initial 10 µs
#endif

// Implicit commit-lease.  Within a certain window after the current
// wrapper was installed, the committing TID holds a soft lease — a
// subsequent negotiate() call from the same TID skips the msec-sleep
// path so it can chain a follow-up commit attempt immediately.  Lease
// auto-expires by wall-clock; no explicit release.  Override via
// -DKAME_PRIORITY_LEASE_DISABLE.
#ifndef KAME_PRIORITY_LEASE_DISABLE
#define KAME_PRIORITY_LEASE
#endif

// --- Debug / assert -------------------------------------------------

// Assert that the given Snapshot/Transaction is NOT currently the
// fair-mode privileged Tx.  Use at any CAS-fail / loop-fail site to
// catch livelock-free invariant violations: a privileged Tx must make
// forward progress, so failing a CAS or re-iterating a spin loop
// while privileged means some other thread bypassed the fair-mode
// yield (= a bug in the negotiate / tag_as_contender coverage).
// Default 0 (production); enable with `-DKAME_STM_ASSERT_PRIVILEGE=1`
// for debug builds.
#ifndef KAME_STM_ASSERT_PRIVILEGE
#define KAME_STM_ASSERT_PRIVILEGE 0
#endif

// Livelock fallback path (kept enabled by default; see negotiate_internal
// comments for the LIVELOCK verdict).
#ifndef KAME_STM_LIVELOCK_FALLBACK
#define KAME_STM_LIVELOCK_FALLBACK 1
#endif

// --- Per-Linkage flip / spin gate subsystem -------------------------

// Master enable for the per-Linkage flip-detection + spin-for-period
// gate subsystem.  When set to 0 (or undefined), the following are
// compiled out:
//   - `ScopedNegotiateLinkage::_neg_spin_block` body + call from
//     `_negotiate_internal`
//   - `Linkage::record_successful_op` body + its two call sites in
//     transaction_impl.h
//   - The Snapshot-side tighten / gate-return latency bookkeeping in
//     `_on_cas_success`
// All the related tuning macros (KAME_KIND_WINDOW_*, KAME_KIND_COUNT_*,
// KAME_SPIN_*, KAME_GATE_RETURN_*) and the RSO_* layout constants are
// still defined unconditionally — they are simply unreferenced when
// the gate is off, so zero runtime/code-size cost beyond the macro
// definitions themselves.
//
// Default: OFF (= 0).  Originally enabled by default on iMac Pro /
// Linux x86 where the per-Linkage spin gate produced a small net
// throughput win on heavily contended bundles.  Re-benchmarked on
// Apple Silicon M3 (2026-05-21, 4-thread payload_integrity / dyn /
// 3level):
//   - 2level: no measurable difference (within noise)
//   - 3level: OFF ~5% faster (mean), lower variance
//   - dyn:    no measurable difference
// On M3 the gate's bookkeeping cost outweighs the saved CV-sleeps,
// so the default is flipped to OFF.  Enable per-target with
// `-DKAME_ENABLE_SPIN_BAND_GATE=1` if a particular workload +
// hardware combination shows a positive A-B result.
#ifndef KAME_ENABLE_SPIN_BAND_GATE
#  if KAME_STM_COMPACT_STATE
//   Compact mode: m_recent_ops_state storage is 32 bits, can't hold the
//   16+16+8+2+22 layout the gate needs. Force off.
#    define KAME_ENABLE_SPIN_BAND_GATE 0
#  else
#    define KAME_ENABLE_SPIN_BAND_GATE 0
#  endif
#endif
#if KAME_STM_COMPACT_STATE && KAME_ENABLE_SPIN_BAND_GATE
#  error "KAME_ENABLE_SPIN_BAND_GATE requires 64-bit lock-free atomics; disable it on this target."
#endif

// Per-Linkage privilege overlay.  When ON (= 1, default), a Tx that
// successfully claims the global fair-mode privilege also walks its
// m_tagged_linkages and CAS-upgrades the kind field of each slot from
// {NONE/BUNDLE/UNBUNDLE} to Reserved (= 3) — peers checking
// `is_priv_stamp` on that Linkage will yield even after the global
// slot has been preempted.  See "Plan A Step 2" commit on the branch.
//
// When OFF (= 0), the claim's per-Linkage CAS walk is compiled out and
// `_fair_blocks` only considers the global slot — i.e. the C++ matches
// the pre-Step-2 (global-only) behaviour.  Useful for A-B benches when
// the per-Linkage overlay is suspected to add overhead without a
// matching throughput gain on a given workload + hardware combination.
//
// Default: ON (= 1).  Disable with -DKAME_PER_LINKAGE_PRIVILEGE=0.
#ifndef KAME_PER_LINKAGE_PRIVILEGE
#define KAME_PER_LINKAGE_PRIVILEGE 1
#endif

// --- Compile-time tuning knobs for the adaptive-negotiate backoff ---

// Half-range of the jittered gate in percent; must be ≥1 (0 causes
// div-by-zero in JITTER_DIV).
#ifndef KAME_STM_JITTER_RANGE
#define KAME_STM_JITTER_RANGE 25
#endif

// Ablation knob: 1 → disable both the jittered gate (gate factor
// pinned to 1.0) and the sleep-chunk ±1ms de-phasing (chunk fixed at
// 1 ms).  For paper comparison / on-off measurement.
#ifndef KAME_STM_DISABLE_JITTER
#define KAME_STM_DISABLE_JITTER 0
#endif

// Gate coefficient: gate opens when mult_wait * GATE_MULT * dt * J < dt2.
// Smaller = more permissive (threads break out sooner).  Default 0
// (closed) — adaptive path supersedes this on the production path.
#ifndef KAME_STM_GATE_MULT
#define KAME_STM_GATE_MULT 1.0f
#endif

// Multiplier on the √C lottery threshold.  1 = ~√C bypass per iteration.
#ifndef KAME_STM_LOTTERY_MULT
#define KAME_STM_LOTTERY_MULT 1
#endif

// Cap on threads simultaneously in the CAS-retry loop.  Positive =
// excess lottery winners fall through to the sleep path to limit
// simultaneous-CAS bursts.  Gate winners (earned priority) are never
// capped.
//  -1 = auto (hardware_concurrency(), fallback to max(C_obs))
//   0 = disabled
//   N > 0 = fixed threshold
#ifndef KAME_STM_MAX_RUNNERS
#define KAME_STM_MAX_RUNNERS 2
#endif

// Floor on concurrent runners; lottery wins are denied while fewer
// than this many runners are active so the wake pipeline has room.
//  -1 = auto, 0 = disabled, N > 0 = fixed
#ifndef KAME_STM_MIN_RUNNERS
#define KAME_STM_MIN_RUNNERS -1
#endif

// --- Adaptive-lease tuning (see lease block in negotiate_internal) -

// Per-Linkage `lease_us` drift schedule.  Asymmetric: growth scales
// with C (capped) so heavy contention climbs quickly; shrink is a
// smaller fixed step so a single C=0 call doesn't undo many C>=2
// adjustments — equilibrium biases toward higher lease where Linkages
// see contention.
#ifndef KAME_LEASE_GROW_PER_C_PERCENT
#define KAME_LEASE_GROW_PER_C_PERCENT 30   // additive per unit of (C-1)
#endif
#ifndef KAME_LEASE_GROW_MAX_PERCENT
#define KAME_LEASE_GROW_MAX_PERCENT   80   // cap on per-call growth
#endif
#ifndef KAME_LEASE_SHRINK_PERCENT
#define KAME_LEASE_SHRINK_PERCENT     5    // shrink step when C == 0
#endif

// Quantized lease write: skip the atomic store unless |delta| ≥ this
// (in µs).  Once the lease pins at MIN/MAX rail, clamping yields
// delta=0 and the store is suppressed — avoids ping-ponging
// m_priority_state.
#ifndef KAME_LEASE_QWRITE_US
#define KAME_LEASE_QWRITE_US 1
#endif

// dt2 fairness gate: suppress owner-skip when the competing tx has
// been waiting longer than this (µs).
#ifndef KAME_DT2_FAIRNESS_US
#define KAME_DT2_FAIRNESS_US 1000
#endif

// Active lease window (us) range, stored per-Linkage as
// Linkage::m_priority_state.lease_us (uint16_t field).
#ifndef KAME_LEASE_US_MIN
#define KAME_LEASE_US_MIN  1     // 1 µs
#endif
#ifndef KAME_LEASE_US_MAX
#define KAME_LEASE_US_MAX  10    // 10 µs — uint16_t field; keep ≤ 65535
#endif

// --- Fair-mode escape thresholds ------------------------------------

// Age-preempt window for the privileged-TID slot: challenger must be
// older than holder by ≥ this many µs to preempt.
#ifndef KAME_STM_PRIV_PREEMPT_WINDOW_US
#  if defined(_WIN32) || defined(WINDOWS) || defined(__WIN32__)
#    define KAME_STM_PRIV_PREEMPT_WINDOW_US 16'000  // 16 ms — Windows timer tick
#  else
#    define KAME_STM_PRIV_PREEMPT_WINDOW_US 1'000   // 1 ms — Linux/macOS
#  endif
#endif

// Floor used for both (a) the LIVELOCK verdict gate (probe fires only
// when tx_age > floor) and (b) the age-preempt threshold in
// try_register_privileged_tidstamp (preemptor must be older than
// holder by floor µs).
#ifndef KAME_STM_PRIV_AGE_NORMAL_US
#  if defined(_WIN32) || defined(WINDOWS) || defined(__WIN32__)
#    define KAME_STM_PRIV_AGE_NORMAL_US 10'000   // 10 ms — Windows scheduler quantum
#  else
#    define KAME_STM_PRIV_AGE_NORMAL_US 300      // 300 us — Linux/macOS
#  endif
#endif

// Maximum extra time a LOW-priority (LOWEST / UI_DEFERRABLE /
// SCRIPTING) privilege holder may keep the slot beyond its initial
// claim threshold.  Once that holder's
// `tx_age > min_privilege_age_us(SCRIPTING) + PRIV_MAX_HOLD_US`,
// the slot is treated as if empty: subsequent
// `try_register_privileged_tidstamp` calls let any thread meeting its
// own claim_floor take over.
//
// **NORMAL / HIGHEST holders are immune** — they are measurement /
// driver critical and must never be disrupted.  The stamp carries
// a 1-bit `lowprio` flag (bit 45 of the packed stamp) set at Tx
// construction based on `getCurrentPriorityMode()`; the expiration
// check gates on that flag.
//
// Without this cap, two stuck SCRIPTING Tx (1 ms claim threshold) can
// deadlock each other forever under the older-only preemption rule
// (a newer SCRIPTING challenger can never preempt an older stuck
// SCRIPTING holder).  This window caps that worst-case starvation.
//
// 50 ms default: combined with SCRIPTING's 1 ms claim floor the
// total expiration window is ~51 ms — within human perception
// budget (~100 ms is the typical "noticeable UI lag" threshold).
// A scripting Tx must never freeze the interactive UI for longer
// than this; if a legitimate low-priority commit needs more time
// it should be restructured into smaller chunks, not granted a
// longer privilege monopoly.
#ifndef KAME_STM_PRIV_MAX_HOLD_US
#define KAME_STM_PRIV_MAX_HOLD_US 50'000   // 50 ms
#endif

// Diagnostic fprintf for priv claim / timeout — default OFF.  Set to
// 1 to enable observability for privilege-hold timeout investigations.
// When ON, every LL-probe successful claim emits one stderr line; this
// disrupts hot-path throughput by ~2-3% on M3 because fprintf locks
// stderr and syscalls, even when the message itself is infrequent.
#ifndef KAME_STM_PRIV_DIAG
#define KAME_STM_PRIV_DIAG 0
#endif

// Per-Priority retry threshold for the livelock probe's verdict (NORMAL
// row; HIGHEST/UI_DEFERRABLE/LOWEST are hard-coded in
// priority_probe_info()).
#ifndef KAME_STM_RETRY_THRESH_NORMAL
#define KAME_STM_RETRY_THRESH_NORMAL 3
#endif

// Floor for the live-contender estimate used in negotiate_internal's
// √C lottery: C_obs = max(C, KAME_STM_C_OBS_MIN).
#ifndef KAME_STM_C_OBS_MIN
#define KAME_STM_C_OBS_MIN 2
#endif

// Legacy √C wake-broadcast lottery in negotiate_internal.  Superseded
// by the per-Linkage spin-for-same-kind path; default 1 (= disabled).
// Set to 0 only for A/B regression.
#ifndef KAME_STM_DISABLE_LOTTERY
#define KAME_STM_DISABLE_LOTTERY 1
#endif

// --- Per-call-site adaptive gate state machine (NegSite::SiteState) -

// Streak-length thresholds for the take_gate FORCE_SLEEP ↔ FORCE_GATE
// transitions, plus the time window for the failure streak validity.
#ifndef KAME_GATE_K_FAIL
#define KAME_GATE_K_FAIL 20             // gate→fail streak depth → FORCE_SLEEP
#endif
#ifndef KAME_GATE_K_SUCC
#define KAME_GATE_K_SUCC 10             // CAS-success streak → FORCE_GATE
#endif
#ifndef KAME_GATE_FAIL_WINDOW_US
#define KAME_GATE_FAIL_WINDOW_US 1000   // 1 ms streak-validity window for fails
#endif
#ifndef KAME_NORMAL_LEASE_US
#define KAME_NORMAL_LEASE_US 50         // 50 µs lease for both FORCE states
#endif

// Legacy per-site adaptive gating (take_gate / FORCE_SLEEP / FORCE_GATE
// tri-state).  Deprecation candidate — superseded by the per-Linkage
// spin-for-same-kind path (KAME_SPIN_* knobs).  Default 0 (= disabled);
// set to 1 only for A/B regression.  KAME_NEGSITE_ENABLED also lights
// up when this is 1 so per-site counters stay live.
#ifndef KAME_LEGACY_GATING
#define KAME_LEGACY_GATING 0
#endif

// ---------------------------------------------------------------------
// Runner digest — peer-readable bit-packed thread state.
//
// Bit-packed snapshot of each thread's negotiation state, published
// per-ScopedNeg to RunnerCounterEntry::digest.  Designed for a future
// "peer-judge" code path that walks the per-thread runner-counter
// linked list (`s_runner_entries_head`) and reads each peer's digest
// to inform CV-sleep decisions.  No consumer exists today.
//
// Default 0 (= digest mechanism compiled out).  K=0 fast-path then
// pays zero atomic writes for it (the publish was a measurable
// per-commit cost on M4 weak-memory at low N).
//
// Set to 1 when adding peer-judge consumers; the mechanism is
// otherwise inert.
#ifndef KAME_ENABLE_RUNNER_DIGEST
#define KAME_ENABLE_RUNNER_DIGEST 0
#endif
#if KAME_STM_COMPACT_STATE && KAME_ENABLE_RUNNER_DIGEST
#  error "KAME_ENABLE_RUNNER_DIGEST requires 64-bit lock-free atomics; disable it on this target."
#endif

// Master switch for the per-call-site adaptive-state machinery
// (`NegSite::Scope`, `m_site_state`, `last_was_gate_return`, the
// state_map hash insert at every `ScopedNegotiateLinkage` ctor, etc.).
// The state is only CONSUMED by one of the four mechanisms below; if
// they are all off, every NegSite write is dead instrumentation.
// Profile (2026-05-21, M3) showed the hash insert path costing ~1.1%
// + the forward_as_tuple plumbing another ~0.8% with all four off.
//
// Default: 1 iff any consumer is enabled; otherwise 0.  Set
// `-DKAME_NEGSITE_ENABLED=1` to keep the state machinery alive (e.g.
// to call `NegSite::dump()` at the end of a run for diagnostics even
// without `KAME_ADAPT_INSTRUMENT`).
#ifndef KAME_NEGSITE_ENABLED
#  if (defined(KAME_LEGACY_GATING) && KAME_LEGACY_GATING) \
   || (defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT) \
   || (defined(KAME_ENABLE_RUNNER_DIGEST) && KAME_ENABLE_RUNNER_DIGEST) \
   || (defined(KAME_ENABLE_SPIN_BAND_GATE) && KAME_ENABLE_SPIN_BAND_GATE)
#    define KAME_NEGSITE_ENABLED 1
#  else
#    define KAME_NEGSITE_ENABLED 0
#  endif
#endif

// --- Spin-for-same-kind / peer-progress path ------------------------

// Hard cap on per-call spin time in µs.  The actual budget is
// min(flip_period_us_ema * KAME_SPIN_BUDGET_PCT / 100, KAME_SPIN_MAX_US).
//
// History: the generic peer-progress spin (waits for ANY change to
// m_transaction_started_time, regardless of peer kind) was previously
// considered structurally bad at high N (spinners delay the holder).
// Earlier sweeps with PCT=100 (1 period budget) and MAX=100 confirmed
// this — all polled / blind variants lost at high N on M4 / M3 Air.
//
// The KAME_SPIN_BUDGET_PCT (period multiplier) and
// KAME_THRASHING_C_MULT (over-thrashing guard) knobs added 2026-05
// reshape the trade-off.  With a wide budget (PCT=600, ~6 periods)
// AND thrashing-skip on a too-short period (period < sig_C * 2),
// the spin path actually beats CV-sleep on both x86 4-core and M4.
//
// MAX=1000 (vs the old 100) accommodates the wider PCT — the cap now
// only kicks in for Linkages with > ~1.6 ms observed period, which is
// effectively never in steady-state contention.
#ifndef KAME_SPIN_MAX_US
#define KAME_SPIN_MAX_US 50
#endif

// ns companion to KAME_SPIN_MAX_US.  Used by `_neg_spin_block`'s
// budget arithmetic which lives entirely in ns to avoid the
// integer-µs underflow at high per-Linkage event counts.  Defaults
// to KAME_SPIN_MAX_US × 1000; override here for sub-µs cap.
#ifndef KAME_SPIN_MAX_NS
#define KAME_SPIN_MAX_NS ((uint64_t)KAME_SPIN_MAX_US * 1000u)
#endif

// Period-multiplier for the spin-budget: spin up to PCT % of the
// observed flip period before yielding to CV-sleep.  600 % was the
// cross-platform sweep winner combined with the C_MULT=2 thrashing
// guard.
#ifndef KAME_SPIN_BUDGET_PCT
#define KAME_SPIN_BUDGET_PCT 600
#endif

// Absolute-time window width (µs) for the windowed per-kind count
// scheme in Linkage::m_recent_ops_state.  Events are bucketed into
// windows of this width, and two adjacent windows (cur + prev) are
// retained.  Counts older than 2 × WINDOW_US are dropped by
// rotation.  Pow-of-2 simplifies the (now_us / WINDOW_US) op.
// WINDOW too short → unidirectional Linkages never accumulate flip
// count, the LOW-band gate never fires, the spin block degenerates to
// 100 % SKIPPED_NO_PERIOD (confirmed via KAME_ADAPT_INSTRUMENT dump
// on the 3level_mixed workload).  128 µs gives ~256 µs visible
// (cur + prev windows) — long enough for a B↔U pair to fall in even
// on lightly contended Linkages, while still short enough that the
// fs_period_ns / count derived spin budget remains in the sub-µs to
// few-µs range we want.
#ifndef KAME_KIND_WINDOW_US
#define KAME_KIND_WINDOW_US 128
#endif

// ns companion to KAME_KIND_WINDOW_US for the spin-budget formula in
// `_neg_spin_block`.  The arithmetic
//   fs_period_ns = (2 * KAME_KIND_WINDOW_NS) / total_count
// stays meaningful at high count (where the µs version underflows to 0).
// The 6-bit `ro_timestamp` packed into m_recent_ops_state is unchanged —
// it remains µs-domain (now_us % 64) and is compared against end_us
// derived from `now_ns() / 1000`.
#ifndef KAME_KIND_WINDOW_NS
#define KAME_KIND_WINDOW_NS ((uint64_t)KAME_KIND_WINDOW_US * 1000u)
#endif

// Gate-return fires only when the per-kind count (summed over the
// cur + prev windows of m_recent_ops_state) falls within the
// sweet-spot band [LOW, HIGH]:
//
//   below LOW   →  not enough evidence the workload is periodic
//                  (= count < threshold semantics from before)
//   in [LOW,HIGH]→  good periodic signal; gate-return safe
//   above HIGH  →  HYPER-thrashing; peer is changing too fast for
//                  our CAS retry to land — fall to CV-sleep,
//                  which naturally de-phases us
//
// The HIGH upper bound replaces the prior "raise the LOW threshold
// on each anti-phase fail" tighten design — pushing LOW up at high
// counts was exactly backwards (hyper-thrashing workloads have high
// counts, and that's PRECISELY where we should NOT gate-return).
// LOW too low (= 3) → spin block enters on Linkages whose flip
// period is too long for the spin to catch, leading to 40-50 %
// TIMEOUT and a wasted spin-then-fall-to-CV-sleep cycle.  LOW too
// high (≥ 12 with WINDOW_US=128) → spin block never enters at all
// — flip counts in a single 2-window observation rarely reach
// double digits even on hot Linkages.
//
// 5 = 5σ confidence floor for Poisson(N): with N flips observed
// per window, σ = √N, so N ≥ 5 puts the signal at 5σ above the
// no-flip noise floor (= count would have to be ≤ 0 by random
// chance) — a reliable "periodic activity is real" threshold.
//
// 3level_mixed N=64, CR=10, 3 s stress sweep (INSTRUMENT):
//   LOW=3  : WON 16.5 % TIMEOUT 41.1 % commits=631 k  ← over-fires
//   LOW=4  : WON 13.9 % TIMEOUT 30.6 % commits=479 k
//   LOW=5  : WON 10.4 % TIMEOUT 24.6 % commits=739 k  ← 5σ pick
//   LOW=6  : WON  8.5 % TIMEOUT 16.2 % commits=858 k
//   LOW=8+ : attempts=0
#ifndef KAME_KIND_COUNT_THRESHOLD
#define KAME_KIND_COUNT_THRESHOLD 3
#endif
// Calibrated to the 16-bit cur_count / prev_count fields in
// m_recent_ops_state (max 65535).  The old default 254 mirrored
// the now-retired 8-bit slot saturation — it would gate out the
// IN_BAND regime entirely on any sustained workload now that
// counts can climb into the thousands.  16383 (= 2^14 - 1) keeps
// the same ~1/4-of-saturation ratio the old 254 / 1023 had on the
// 10-bit slots, with `tighten` right-shifts still giving useful
// adaptive narrowing.
#ifndef KAME_KIND_COUNT_HIGH
#define KAME_KIND_COUNT_HIGH 16383
#endif

// Adaptive band narrowing on anti-phase fails.  Each detected fail
// (Snapshot::m_last_gate_returned still true at next gate-return
// decision) increments m_gate_return_tighten, which RIGHT-SHIFTS the
// effective HIGH bound:
//
//   effective_high = KAME_KIND_COUNT_HIGH >> tighten
//
// At tighten = log2(HIGH / LOW), the band collapses to [LOW, LOW] —
// only an exact-count match fires; beyond, the gate is effectively
// closed.  Reset to 0 on any CAS success.
#ifndef KAME_GATE_RETURN_MAX_TIGHTEN
#define KAME_GATE_RETURN_MAX_TIGHTEN 4
#endif

#endif /* TRANSACTION_DEFINITIONS_H */
