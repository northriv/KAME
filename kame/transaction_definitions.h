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
// adaptive-backoff machinery. Every macro here is `-D`-overridable at
// cmake time (or via the toolchain CXXFLAGS). Header is included from
// the top of transaction.h so the declarations of Node<XN>::Linkage /
// NegotiationCounter / ScopedNegotiateLinkage can reference them.
//
// Defaults reflect the most recent sweep winners on iMac Pro / Apple
// Silicon; see git history for rationale per knob.
// =====================================================================

// Initial per-Linkage lease (ns). Stored as µs in the packed priority
// state (uint16_t field), so the runtime default of 10000 ns = 10 µs.
#ifndef KAME_LEASE_NS_BASE
#define KAME_LEASE_NS_BASE 10000    // initial 10 µs
#endif

// Implicit commit-lease. within a certain window after the
// current wrapper was installed, the committing TID holds a soft lease —
// a subsequent negotiate() call from the same TID skips the msec-sleep
// path so it can chain a follow-up commit attempt immediately. Lease
// auto-expires by wall-clock; no explicit release. Override via
// -DKAME_PRIORITY_LEASE_DISABLE.
#ifndef KAME_PRIORITY_LEASE_DISABLE
#define KAME_PRIORITY_LEASE
#endif

// Assert that the given Snapshot/Transaction is NOT currently the
// fair-mode privileged Tx. Use at any CAS-fail / loop-fail site to
// catch livelock-free invariant violations: a privileged Tx must
// make forward progress, so failing a CAS or re-iterating a spin
// loop while privileged means some other thread bypassed the
// fair-mode yield (= a bug in the negotiate / tag_as_contender
// coverage). Default 0 (production); enable with
// `-DKAME_STM_ASSERT_PRIVILEGE=1` for debug builds.
#ifndef KAME_STM_ASSERT_PRIVILEGE
#define KAME_STM_ASSERT_PRIVILEGE 0
#endif

// Livelock fallback path (kept enabled by default; see negotiate_internal
// comments for the LIVELOCK verdict).
#ifndef KAME_STM_LIVELOCK_FALLBACK
#define KAME_STM_LIVELOCK_FALLBACK 1
#endif

// --- Compile-time tuning knobs for the adaptive-negotiate backoff ---
// All are -D overridable at cmake time.

// Half-range of the jittered gate in percent; must be ≥1 (0 causes div-by-zero
// in JITTER_DIV). Sweep (N=128 median/3): JIT=10 avg4=4841k > JIT=25 avg4=4672k.
#ifndef KAME_STM_JITTER_RANGE
#define KAME_STM_JITTER_RANGE 10
#endif

// Ablation knob: 1 → disable both the jittered gate (gate factor pinned to
// 1.0) and the sleep-chunk ±1ms de-phasing (chunk fixed at 1 ms). For paper
// comparison / on-off measurement.
#ifndef KAME_STM_DISABLE_JITTER
#define KAME_STM_DISABLE_JITTER 0
#endif

// Gate coefficient: gate opens when mult_wait * GATE_MULT * dt * J < dt2.
// Smaller = more permissive (threads break out sooner). Default 0 (closed).
#ifndef KAME_STM_GATE_MULT
#define KAME_STM_GATE_MULT 1.0f
#endif

// Multiplier on the √C lottery threshold. 1 = ~√C bypass per iteration.
#ifndef KAME_STM_LOTTERY_MULT
#define KAME_STM_LOTTERY_MULT 1
#endif

// Cap on threads simultaneously in the CAS-retry loop.
// Positive = excess lottery winners fall through to the sleep path to
// limit simultaneous-CAS bursts. Gate winners (earned priority) are
// never capped.
//  -1 = auto (hardware_concurrency(), fallback to max(C_obs))
//   0 = disabled
//   N > 0 = fixed threshold
#ifndef KAME_STM_MAX_RUNNERS
#define KAME_STM_MAX_RUNNERS 2   // sweep winner: 2 > 10 > -1 > 6 > 8 > 4 > 16
#endif

// Floor on concurrent runners; lottery wins are denied while fewer
// than this many runners are active so the wake pipeline has room.
//  -1 = auto (hardware_concurrency(), fallback to max(C_obs))
//   0 = disabled
//   N > 0 = fixed threshold
#ifndef KAME_STM_MIN_RUNNERS
#define KAME_STM_MIN_RUNNERS -1
#endif

// --- Adaptive-lease tuning (see lease block in negotiate_internal) ---
// Per-Linkage `lease_us` drifts by these schedules. Asymmetric: growth
// scales with C (capped) so heavy contention climbs quickly; shrink is
// a smaller fixed step so a single C=0 call doesn't undo many C>=2
// adjustments — equilibrium biases toward higher lease where Linkages
// see contention.
#ifndef KAME_LEASE_GROW_PER_C_PERCENT
#define KAME_LEASE_GROW_PER_C_PERCENT 30   // additive per unit of (C-1)
#endif
#ifndef KAME_LEASE_GROW_MAX_PERCENT
#define KAME_LEASE_GROW_MAX_PERCENT   80   // cap on per-call growth
#endif
#ifndef KAME_LEASE_SHRINK_PERCENT
#define KAME_LEASE_SHRINK_PERCENT     10   // shrink step when C == 0
#endif
// Quantized lease write: skip the atomic store unless |delta| ≥ this
// (in µs). Once the lease pins at MIN/MAX rail, clamping yields delta=0
// and the store is suppressed — avoids ping-ponging m_priority_state.
#ifndef KAME_LEASE_QWRITE_US
#define KAME_LEASE_QWRITE_US 1
#endif
// dt2 fairness gate: suppress owner-skip when the competing tx has
// been waiting longer than this (µs). Hot-loop commits (dt2 ≈ 2-8 µs)
// stay below the gate and benefit from the skip; long-held tx
// (test_negotiation msecsleep, dt2 ≈ 60-150 µs) bypass it.
#ifndef KAME_DT2_FAIRNESS_US
#define KAME_DT2_FAIRNESS_US 2000
#endif

// Floor used for both (a) the LIVELOCK verdict gate (probe fires only
// when tx_age > floor) and (b) the age-preempt threshold in
// try_register_privileged_tidstamp (preemptor must be older than holder
// by floor µs). Values balance "fire quickly when stuck" vs "don't
// thrash on normal contention". 20 ms for HIGHEST/NORMAL was selected
// to avoid the dynamic_node_test churn-deadlock seen at 1ms (short
// Txs aged past 1ms trivially → constant preempt-cycle).
#ifndef KAME_STM_PRIV_AGE_NORMAL_US
#define KAME_STM_PRIV_AGE_NORMAL_US 20'000   // 20 ms — sweep winner (≥20ms for RT safety)
#endif

// Per-Priority retry threshold for the livelock probe's verdict
// (NORMAL row; HIGHEST/UI_DEFERRABLE/LOWEST are hard-coded in
// priority_probe_info()).
#ifndef KAME_STM_RETRY_THRESH_NORMAL
#define KAME_STM_RETRY_THRESH_NORMAL 4   // sweep winner: 4 > 3 > 5 at AGE=20ms
#endif

// Adaptive-lease constants. The active lease window (us) is stored per-Linkage
// as Linkage::m_priority_state (lease_us field), so contention sites converge
// independently (hot Linkage → longer lease; cold Linkage → short).
// Clamped to [KAME_LEASE_US_MIN, KAME_LEASE_US_MAX].
// Fixed-threshold drift is used instead of proportional-rate variants
// because the C distribution is heavily skewed toward low C; keeping
// C == 1 neutral avoids dragging the lease down during the low-C phase
// that dominates total call count.
#ifndef KAME_LEASE_US_MIN
#define KAME_LEASE_US_MIN  1     // 1 µs
#endif
#ifndef KAME_LEASE_US_MAX
#define KAME_LEASE_US_MAX  3    // 3 µs — uint16_t field; keep ≤65535. Sweep winner (MAX_R=2).
#endif

// Lottery firing at the wake-broadcast point. Default: blocking
// lock_guard for reliable wakes. Rebuild with -DKAME_STM_NOTIFY_TRY_LOCK=1
// to select the try_lock skip variant for ablation / regression measurement.
#ifndef KAME_STM_DISABLE_LOTTERY
#define KAME_STM_DISABLE_LOTTERY 0
#endif

#endif /* TRANSACTION_DEFINITIONS_H */
