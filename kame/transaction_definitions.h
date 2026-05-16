/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef TRANSACTION_DEFINITIONS_H
#define TRANSACTION_DEFINITIONS_H

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
#define KAME_DT2_FAIRNESS_US 4000
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

// Per-Priority retry threshold for the livelock probe's verdict (NORMAL
// row; HIGHEST/UI_DEFERRABLE/LOWEST are hard-coded in
// priority_probe_info()).
#ifndef KAME_STM_RETRY_THRESH_NORMAL
#define KAME_STM_RETRY_THRESH_NORMAL 4
#endif

// Floor for the live-contender estimate used in negotiate_internal's
// √C lottery: C_obs = max(C, KAME_STM_C_OBS_MIN).
#ifndef KAME_STM_C_OBS_MIN
#define KAME_STM_C_OBS_MIN 2
#endif

// √C wake-broadcast lottery in negotiate_internal.  Probabilistic
// per-iteration bypass meant to break tied retry loops; predates the
// per-Linkage spin-for-same-kind path.
//
// Default 1 (= disabled) since the spin path now handles the same
// pivot more deterministically — and at N ≥ 8 the lottery becomes
// actively counter-productive (it wakes O(C) threads on every
// notify_n_contenders call, multiplying the post-wake CAS retry
// storm).  Set to 0 to restore the legacy probabilistic lottery for
// A/B regression.
//
// 4-thread Linux x86 sweep (KAME_DISABLE_TAKE_GATE_RETURN=1,
//   3-run median, 2-second stress mode, total commits over 2 s):
//     N=4       +0 / +2 %   (noise)
//     N=8      +6 /  +5 %
//     N=16    +10 / +11 %
//     N=32    +29 / +18 %
//     N=64    +59 / +55 %     ← peak gain (16× oversubscription)
//     N=128   +42 /  -6 %     ← N=128 3L: variance-dominated
//   Pattern: linear improvement up to ~16× oversubscription, then
//   tapering as scheduler overhead overtakes algorithm gain.  Mirrors
//   reports on Apple Silicon M4 (user note: "M4 では速くなる").
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

// ---------------------------------------------------------------------
// Legacy per-site adaptive gating.  Deprecation candidate.
//
// The take_gate / FORCE_SLEEP / FORCE_GATE tri-state machine was the
// pre-spin-path mechanism for bypassing the negotiate sleep on
// heavy-thrash sites.  Per-site streak counters (consec_succs /
// consec_fails) drove FORCE state transitions; `take_gate` was
// consumed by an early `break` in negotiate_internal's gate-decision
// block, and `_on_cas_success` / `_on_cas_fail` updated the streak
// counters and triggered state changes.
//
// The newer per-Linkage spin-for-same-kind path (see KAME_SPIN_* knobs
// above) supersedes the gate-return shortcut with finer granularity
// (per-Linkage, not per-call-site), µs-bounded wait, and peer-progress
// awareness via m_transaction_started_time observation.  The two
// mechanisms conflict directly: take_gate FORCE_GATE intercepts before
// the spin block, so spin never gets a chance on the sites it was
// designed for.
//
// Default 0 (= legacy gating disabled).  When 0, the gate-decision
// state machine and all driving streak / state-transition logic in
// _on_cas_success / _on_cas_fail are wrapped out — control flows
// straight through the lottery → spin → sleep cascade.  Per-site
// INSTRUMENT counters that don't depend on take_gate (entries,
// commits, blocked_by_peer / gate_returns_by_peer seen on the path)
// continue to update for diagnostic dump.
//
// Set to 1 to restore the legacy gating mechanism for A/B regression
// or for workloads where it might still help (e.g. very-high
// scope-affinity site patterns).  When 1, all gating-tagged code is
// reactivated; no other knob changes are needed.
//
// Sweep, 4-thread Linux x86, 3-run median, 2-second stress mode,
// total commits / 2 s (gating off vs on):
//     N=4       +0 / +2 %   (noise)
//     N=8      +6 /  +5 %
//     N=16    +10 / +11 %
//     N=32    +29 / +18 %
//     N=64    +59 / +55 %     ← peak gain (16× oversubscription)
//     N=128   +42 /  -6 %     ← N=128 3L: variance-dominated
//   Pattern: linear improvement up to ~16× oversubscription, then
//   tapering as scheduler overhead overtakes algorithm gain.  Mirrors
//   reports on Apple Silicon M4 (user note: "M4 では速くなる").
#ifndef KAME_LEGACY_GATING
#define KAME_LEGACY_GATING 0
#endif

// ---------------------------------------------------------------------
// Runner digest — peer-readable bit-packed thread state.
//
// Bit-packed snapshot of each thread's negotiation state, published
// per-ScopedNeg to RunnerCounterEntry::digest.  Designed for a future
// "peer-judge" code path that walks s_runner_counters and reads each
// peer's digest to inform CV-sleep decisions.  No consumer exists
// today.
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
#define KAME_SPIN_MAX_US 1000
#endif

// (Retired) KAME_SPIN_RECENT_FLIP_US — used to gate the spin block
// via `age > RECENT` SKIPPED_COLD outcome.  Removed when the
// unified PRE-spin band gate replaced the age check: spin entry is
// now decided by the per-kind count in the [LOW, HIGH] window
// (m_recent_ops_state), not by a wall-clock age.  Macro kept here
// solely so legacy -D flags don't error; it has no effect on
// behaviour.

// --- Kind-match coalesce spin (experimental, opt-in) ---------------
//
// Tighter spin variant that fires only when the current slot holder
// (peer) is performing the *same kind* of operation as us.  Rationale:
// when peer_kind == my_kind, the slot will be released into a state we
// can step into directly; we bet on being next in line.  When the
// kinds differ, spinning has no payoff (we'd retry the CAS only to
// lose to a different kind anyway), so we skip straight to negotiate_sleep.
//
// Mode selector:
//   0 = OFF (legacy any-change spin path active iff KAME_SPIN_RECENT_FLIP_US > 0)
//   1 = K1 polled strict — win on slot release; kind change = lose (exit spin)
//   2 = K2 polled loose  — win on slot release OR kind change
//   4 = K4 blind         — no in-loop reads (ARM stlxr-friendly)
//
// Budget = min(flip_period_ema, KAME_COALESCE_MAX_US).
// Entry condition (all required):
//   my_kind != NONE
//   peer_kind == my_kind
//   age <= KAME_COALESCE_RECENT_US
//   ops_since_flip < sig_C * 8
//
// When MODE != 0, the legacy KAME_SPIN_RECENT_FLIP_US path is bypassed
// completely — the coalesce trigger replaces it.
//
// macOS M4 sweep (2026-05):
//   Phase 1 (3 modes @ MAX=50 RECENT=200):
//     MODE=0 spin OFF baseline:       avg8 = 5 455 165
//     MODE=4 K4 blind:                avg8 = 5 168 946  (−5.25%)
//     MODE=2 K2 loose:                avg8 = 5 030 149  (−7.79%)
//     MODE=1 K1 strict:               avg8 = 5 016 120  (−8.04%)
//     K4 > K2 > K1: ARM stlxr interference is real (~3% gap K1→K4)
//     but the dominant cost is the spin trigger firing on every
//     contended commit (kind-match almost always true).
//
//   Phase 2 (K4 parameter tightening):
//     MAX=50 RECENT=200 (initial):    avg8 = 5 168 946  (−5.25%)
//     MAX=30 RECENT=100:              avg8 = 5 287 101  (−3.08%)
//     MAX=10 RECENT=200:              avg8 = 5 293 740  (−2.96%)
//     MAX=10 RECENT=50:               avg8 = 5 403 890  (−0.94%)
//     MAX=10 RECENT=30:               avg8 = 5 283 464  (−3.15%)
//     MAX=5  RECENT=50:               avg8 = 5 308 442  (−2.69%)
//     MAX=5  RECENT=30  ← winner:     avg8 = 5 426 038  (−0.53%)  ← within noise
//     N128CR2_3L recovers from −10.96% (M10R50) to +0.92% at M5R30.
//     N128CR10_3L remains the persistent regressor (~−6%).
//
// Conclusion: with M5R30 defaults, K4 is statistically tied with
// spin OFF on M4.  The opt-in is preserved (MODE=0 default) so the
// shipping behaviour is unchanged.  Defaults below match the M4
// sweep winner so enabling MODE=4 alone gets the tuned config.
#ifndef KAME_COALESCE_MODE
#define KAME_COALESCE_MODE 0
#endif

// Hard cap on coalesce spin budget (µs).  Actual budget is
// min(flip_period_ema, KAME_COALESCE_MAX_US).
#ifndef KAME_COALESCE_MAX_US
#define KAME_COALESCE_MAX_US 5
#endif

// Trigger gate (µs): spin only if last flip was within this window.
// Tight value keeps the spin from firing on stale "recently active"
// linkages that have since cooled.
#ifndef KAME_COALESCE_RECENT_US
#define KAME_COALESCE_RECENT_US 30
#endif

// Spin-budget scaling, expressed as a percentage of fs_period_us.
// budget = min(fs_period * PCT / 100, KAME_COALESCE_MAX_US).
//   100 = exactly one observed flip period (default for polled modes
//         K1 / K2, which can early-exit so over-shooting is cheap)
//    75 = 3/4 period (default for K4 blind; can't early-exit so a
//         tighter budget caps the worst-case waste)
//   200 = two periods (for x86 / wide-core sweeps where waiting two
//         observed cycles further raises the same-kind-coalesce hit
//         rate)
// Set explicitly to override per-mode defaults.
#ifndef KAME_COALESCE_BUDGET_PCT
#  if KAME_COALESCE_MODE == 4
#    define KAME_COALESCE_BUDGET_PCT 75
#  else
#    define KAME_COALESCE_BUDGET_PCT 100
#  endif
#endif

// Same knob for the legacy any-change spin path
// (KAME_SPIN_RECENT_FLIP_US > 0).  Always polled, so over-shooting is
// cheap (early-exit on any slot change).
//
// macOS M4 sweep (2026-05, RECENT=1000 MAX=1000 C_MULT=2):
//   PCT=300:  avg8 ≈ 5.40 M  (mild gain on x86 noise, loss on M4 3L)
//   PCT=600:  avg8 = 5 468 017  ← winner (+0.24% vs spin OFF)
//   PCT=800:  avg8 = 5 446 509  (essentially tied)
//   PCT=1000: avg8 ≈ 5.40 M  (over-spin)
// Same PCT family was the cross-platform winner on x86 4-core (the
// long-period sweep that motivated the knob).  600 % = spin up to
// 6× the observed flip period before yielding to CV-sleep; combined
// with C_MULT=2 thrashing guard this turns out positive at all N on
// M4 and on x86 4-core.
#ifndef KAME_SPIN_BUDGET_PCT
#define KAME_SPIN_BUDGET_PCT 600
#endif

// Over-thrashing detection multiplier for the SKIPPED_THRASHING gate.
// Spin is skipped iff
//   fs_period_us * KAME_THRASHING_C_MULT_DEN
//     < sig_C * KAME_THRASHING_C_MULT
// i.e. effective ratio = KAME_THRASHING_C_MULT / KAME_THRASHING_C_MULT_DEN.
// Use DEN to express non-integer multipliers (phase effects in B/U
// cycles can put the optimum between consecutive integers):
//   NUM=2, DEN=1 → ratio 2.0  (default; "1 round-robin + 1 period margin")
//   NUM=5, DEN=2 → ratio 2.5  (mid-point between C=2 and C=3 sweep peaks)
//   NUM=3, DEN=1 → ratio 3.0
// Larger ratio → more cases classified as THRASHING → more gate-return
// firings on peer same-kind, fewer spin attempts.
#ifndef KAME_THRASHING_C_MULT
#define KAME_THRASHING_C_MULT 2
#endif
#ifndef KAME_THRASHING_C_MULT_DEN
#define KAME_THRASHING_C_MULT_DEN 1
#endif

// Absolute-time window width (µs) for the windowed per-kind count
// scheme in Linkage::m_recent_ops_state.  Events are bucketed into
// windows of this width, and two adjacent windows (cur + prev) are
// retained.  Counts older than 2 × WINDOW_US are dropped by
// rotation.  Pow-of-2 simplifies the (now_us / WINDOW_US) op.
#ifndef KAME_KIND_WINDOW_US
#define KAME_KIND_WINDOW_US 256
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
#ifndef KAME_KIND_COUNT_THRESHOLD
#define KAME_KIND_COUNT_THRESHOLD 2
#endif
#ifndef KAME_KIND_COUNT_HIGH
#define KAME_KIND_COUNT_HIGH 32
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

// Slot release strategy.  When a Tx commits / cleans up, it currently
// zero-stores its tag from each tagged Linkage's
// m_transaction_started_time slot (drop_tags_n_privilege, line ~1747
// of transaction.h).
//
// KAME_SLOT_KEEP_KIND=1 changes the release to leave the kind bits
// behind ("0 us + my_kind") so that subsequent peer readers can see
// "the last released kind on this Linkage", extending the
// same-kind-coalesce hint window from "peer currently holding" to
// "peer just released with our kind".
//
// All slot==0 checks (Linkage::negotiate fast path, livelock probe
// match, spin-loop release detection, etc.) are routed through
// `NegotiationCounter::is_active_stamp(s)` so the kind-only state
// is recognised as "no active tagger" — semantic invariants
// preserved.
//
// Default 0 (off): a clean experiment knob.  Enabling adds a single
// `stamp_us(load) != 0` shift on the negotiate fast path; the
// expected gain comes from gate-return firing on a wider time window
// (peer-just-released also matches), at the cost of one extra mask
// on the cold path.
#ifndef KAME_SLOT_KEEP_KIND
#define KAME_SLOT_KEEP_KIND 0
#endif

#endif /* TRANSACTION_DEFINITIONS_H */
