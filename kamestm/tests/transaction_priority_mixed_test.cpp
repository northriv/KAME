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
//! HIGHEST + NORMAL + UI_DEFERRABLE mixed-priority livelock hunt.
//!
//! Motivated by a field report: KAME livelocks rarely when the UI is operated
//! during an NMR measurement, suspicion on the HIGHEST-ification.  This test
//! reproduces that deployment's ROLES, not just its thread counts:
//!
//!   acquisition  ONE thread oscillating exactly like finishWritingRaw: the
//!                record commit on its driver subtree at HIGHEST, then
//!                ScopedDemoteRealtime + the 20 ms ScopedWaitBudget for the
//!                demoted downstream (entry writes on the same subtree and a
//!                visualize-ish snapshot) at NORMAL, then back.  The
//!                HIGHEST<->NORMAL oscillation on one thread is the real
//!                deployment shape — a first draft used a separate
//!                always-NORMAL downstream thread, which is both harsher and
//!                wrong (it never self-throttles the HIGHEST churn).
//!   NORMAL       other drivers' threads on their own subtree.
//!   UI           UI_DEFERRABLE, the main-thread mix: frequent ROOT Snapshots
//!                (graph redraws bundle the root and absorb the driver
//!                packets — the documented always-fail shape for descendant
//!                commits), leaf widget writes, occasional root-scope
//!                transactions, structural insert/release churn (tool
//!                creation), and — the typical NMR trigger — writes into the
//!                MEASURING driver's own subtree (changing averaging etc.
//!                mid-acquisition).
//!
//! Reading the OS arm's results (measured 2026-08, 4-CPU x86-64, non-RT
//! kernel, so treat the numbers as shape rather than as a bound).
//!
//! **All of the figures in this block were taken at KAME_MIX_RT_POOL=0**, i.e.
//! before the realtime contract became the default — see that knob for the
//! measurement that changed it and by how much.  Re-measure before comparing
//! anything here against a run at today's defaults:
//!
//!   * The clean 2x2, on a PREEMPT_RT host (i5-7500, isolcpus=2,3), acq
//!     commits/s: neither 146.9k; FIFO+pin only 155.0k; XSUBTREE only 89.4k;
//!     both 57.8k.  So **FIFO+pin costs nothing** (+6 %), the cross-subtree
//!     role costs **1.64x** on its own — that is the role's whole point,
//!     since without it the NORMAL peers only ever touch their own subtree and
//!     cannot contend with the acquiring driver at all — and the two together
//!     cost **2.54x**, well past the 1.54x their product predicts.  Something
//!     is super-additive.
//!   * That "something" is NOT the isolation.  Same knobs, same two-CPU shape,
//!     only the acquisition core changed: `taskset -c 0,1` (both housekeeping)
//!     gave 49.8k, `taskset -c 0,3` (onto the `nohz_full` isolated core) gave
//!     **53.9k** — the isolated core is marginally *faster*, so the
//!     wake-a-tickless-core hypothesis is refuted.  What is left is ordinary
//!     SMP: pinning forces the cross-subtree contention to be cross-core on
//!     every single conflict, where an unpinned CFS may co-locate the two
//!     threads and settle some conflicts in one cache.  Nothing RT-specific,
//!     and nothing that argues against isolating the acquisition core.
//!     (Read this as the THROUGHPUT effect it is, at the 20 ms budget these
//!     runs used.  The latency picture below is the opposite way round:
//!     isolating the acquisition core is what removes the budget-exempt tail,
//!     and it is not optional.)
//!   * Note in passing that cramming the three housekeeping threads onto one
//!     core made them *collectively faster* (1.22M vs 911k commits/s spread
//!     over four), which is the same coherence effect seen from the other
//!     side.
//!   * **All of the above is the HIGHEST arm, which KAME does not ship.**
//!     `XPrimaryDriverWithThread::AcquisitionPriority` is
//!     `ScopedPriority(NORMAL)` plus an OS elevation — the kamestm HIGHEST tier
//!     was retired for KAME because per-record analyses cannot honour its
//!     precondition — and the difference is not a matter of degree.  HIGHEST
//!     cannot sleep (`if(entry_pr == Priority::HIGHEST) break;` sits at the TOP
//!     of the negotiator's round loop, above both negotiate_sleep call sites);
//!     NORMAL sleeps in 1-2 ms chunks.  Same host, same knobs, only the tier:
//!     p50 unchanged at 768 ns, p99 1.28 us against 2.05, then p99.9 3.67 ms
//!     against 20.5 us (179x) and MAX 20.19 ms against 95.1 us (212x).  The
//!     other roles completed a mean of 2,004 commits during each slow one
//!     against 13 in the HIGHEST arm: a thread asleep while the system works.
//!     SCHED_FIFO changed nothing in THIS measurement (43.8k/s and MAX 20.15 ms
//!     with, 43.3k/s and 20.19 ms without) — but do not read the inference that
//!     came with it ("no scheduling class shortens a voluntary wait"), because
//!     it is false.  Both arms are budget-dominated at 20 ms, so a wake-up cost
//!     of a few hundred microseconds cannot show up in either MAX; and the two
//!     arms are not the same measurement at that scale anyway, since Linux
//!     applies 50 us of timer slack to every futex timeout a SCHED_OTHER task
//!     arms and none to an RT one.  On the wake-up cost itself the scheduling
//!     class is the DOMINANT term (5.3x) — see the correction below.
//!   * So at NORMAL the wait budget is the only bound the record commit has,
//!     and — pinned — it delivers exactly its value.  Sweep on the RT host,
//!     FIFO + pin, 60 s each:
//!
//!         budget   commits/s     mean       MAX    MAX-budget   clipped
//!         2000 us     76,904   7991 ns   2.179 ms     179 us     0.333 %
//!         1000 us    131,721   4101 ns   1.223 ms     223 us     0.320 %
//!          500 us    185,700   2409 ns   0.662 ms     162 us     0.304 %
//!          200 us    251,933   1534 ns   0.408 ms     208 us     0.334 %
//!
//!     MAX = budget + a constant ~200 us of overshoot, with NO floor down to
//!     200 us, and throughput RISES 3.3x as the budget falls because a clipped
//!     commit stops sleeping and retries.  The clip rate is invariant at
//!     ~0.32 %: the same population of commits is caught, just earlier and
//!     cheaper.  (XPrimaryDriver::downstreamWaitBudgetUS()'s documented "20 ms
//!     costs 4.7 % of throughput" came from the grand-scope 8-thread arm and
//!     does NOT hold here — here smaller is better on both axes.)
//!     Confirmed at length: 300 s, FIFO + pin, 1 ms budget, **38,303,308
//!     commits, MAX 1.288 ms, ZERO over a 3 ms deadline**, with every other
//!     role healthy (UI 42.3k/s, SCRIPTING 129.7k/s, NORMAL 100.6k/s).
//!   * **CORRECTION to the table above, from the NegDiag instrumentation this
//!     file now carries (see SlowDiag).**  The ~200 us was real and the
//!     attribution was wrong.  It is not the budget-exempt wait: `rounds_exempt`
//!     is ZERO across 17,274 slow commits, under every scheduling class,
//!     C-state setting and budget tried.  It is the timed wait's own WAKE-UP
//!     cost — the worst commit of a 200 us-budget run is one cell.wait() asked
//!     for 198,000 ns that returned 695,599 ns later, with 6,195 ns unaccounted,
//!     and that remainder is the STM's entire share.  Decomposed as the worst
//!     single overshoot: plain 662 us, PM-QoS alone 605, FIFO alone 124,
//!     PM-QoS + FIFO 20 — the scheduling class dominates at 5.3x and PM-QoS
//!     buys its 6x only on top of it, super-additively, which is why the
//!     earlier single-knob arms concluded the residue was irreducible.
//!     Fixed at the source: a budgeted sleep now stops KAME_NEG_SPIN_TAIL_US
//!     (300) short of the deadline and polls the remainder, so MAX - budget is
//!     7.1 us at 20 ms and 3.0 us in the ship configuration — below this host's
//!     own 17 us floor.  The sweep above is therefore PRE-RESERVE data; its
//!     shape (linear in the budget, throughput rising as it falls, clip rate
//!     invariant) still holds, its constant does not.  Note the cliff: once the
//!     reserve approaches the whole budget the thread never sleeps and starves
//!     the deferrable tiers (-94 %/-98 % at a 200 us budget), so keep the budget
//!     well above 300 us.
//!   * **Pinning is what makes the budget work, and FIFO without it is
//!     catastrophic.**  Unpinned, MAX sticks at 12-13 ms for every budget from
//!     5 ms down to 500 us while the clip count saturates.  The standing
//!     explanation is the wait behind a LIVE PRIVILEGED PEER, which is
//!     contractually budget-exempt and therefore bounded by the holder's
//!     completion time rather than by the budget — but treat it as a HYPOTHESIS,
//!     because it is the same hypothesis the instrumentation above just refuted
//!     for the pinned arm, and nobody has yet run the instrumented build
//!     UNPINNED.  `rounds_exempt` over an unpinned arm settles it in one run;
//!     until then the 12-13 ms could equally be more late wake-ups.  Pinning
//!     the contenders together (acq alone on cpu3, everyone else on cpu0) does
//!     make a holder promptly scheduled among its peers, and the residue
//!     disappears entirely.  Take the isolation away while keeping FIFO
//!     and the same run FAILS the watchdog: only the acq thread is put on
//!     SCHED_FIFO here, so it preempts the very CFS holders it then waits
//!     behind — UI fell to 144 commits/s and SCRIPTING to 176 (from 42.3k and
//!     129.7k), both flagged at 6,001 ms, while acq ran away at 337k/s and
//!     still took 50.9 ms on its own worst commit.  A classic priority
//!     inversion, and the reason `isolcpus` is not optional: **FIFO and
//!     isolation ship together or neither ships.**
//!   * **Where the HIGHEST tail actually is, from the SlowDiag block below**
//!     (RT host, isolation with the tick verified stopped, with_pmqos, contract
//!     honoured, 120 s, slow threshold 15 us, n = 8,007): the negotiator
//!     accounts for NONE of it.  sleeps 0.00, spins 0.00, slept 0 ns, spin
//!     0 ns, exempt rounds 0.00, worst single wait overshoot 0 ns — and in the
//!     worst commit taken whole, 51,796 ns of 51,796 ns unaccounted.  `rounds`
//!     is 1.29 per commit against `entries` 4.58, i.e. most entries return
//!     above the round loop.  The thread is not waiting for anybody; the time
//!     is inside commit().
//!     KAME_MIX_PHASE then measured it instead of inferring it.  Over 9,172
//!     slow commits: snapshot 947 ns, payload write 1,364 ns, the SUCCESSFUL
//!     commit 1,199 ns, and failed attempts plus their re-snapshot 15,778 ns.
//!     The worst commit, 50,707 ns, splits 729 / 2,591 / 669 / 46,670 with
//!     48 ns unattributed — the accounting closes.  A failing attempt costs
//!     ~13x a succeeding one, which is where the arithmetic below pointed.
//!     (A container said the payload write instead; its floor produces 95 us
//!     events and a machine stall lands in whichever phase is running, which
//!     is preferentially the longest.  Phase attribution needs the 219 ns
//!     floor to mean anything.)
//!     The arithmetic that predicted it: that worst commit took 4 attempts
//!     over 8 entries — two linkages per attempt, which is the root->devA path
//!     — at ~13 us per attempt against 823 ns for a clean commit, so a LOSING
//!     attempt costs ~15x a winning one.  Bundle/unbundle done and discarded
//!     fits every other arm: only a peer whose transaction SPANS the acquiring
//!     subtree provokes it (5x), the cost is path-shaped so 4x the leaves
//!     leaves the magnitude alone, and a root Snapshot at 42 kHz — which
//!     bundles but does not span — does nothing.  Note that this does not
//!     contradict the UI finding below: a snapshot's bundle and a spanning
//!     transaction's unbundle are different events, and only the second is on
//!     the acquiring thread's path.
//!     BUT IT DOES NOT FIT THE MAGNITUDE, and that was never checked before it
//!     went into the docs.  Multiply it out: 2 entries per failed attempt x
//!     bundle+unbundle, each bounded ABOVE by the whole successful commit
//!     phase (1,199 ns, which contains a bundle), is 4.8 us against 15.6 us
//!     measured — short by 3.2x, or 6.5x counting bundle alone.  And
//!     bundle_cas_retries is 0.00, so it is not spinning its way there either.
//!     No counter could settle it because nothing timed the pass.
//!     Now one does: bundle() and unbundle() are timed (outermost call only —
//!     both recurse) and differenced across the SAME boundaries as `retry`, so
//!     the split needs no arithmetic.  Measured on the RT host, isolated pair,
//!     FIFO + pinning, 14.3 M commits: **40 % of the retry phase on the mean,
//!     24 % in the worst commit** (7,133 of 30,245 ns); the housekeeping-pair
//!     run says 36/29 % and a container 30-32 %, so 30-40 % everywhere.
//!     Re-bundling is a real and substantial term and it is not the mechanism;
//!     ~two thirds of the cost of a failed attempt remain unattributed, and the
//!     next candidate has to clear the same multiplication this one failed.
//!     That same run is the cleanest statement of the finding as a whole:
//!     MAX 40,812 ns = 998 snapshot + 1,588 write + 7,925 successful commit +
//!     30,245 retry, 56 ns unattributed, in a run with ZERO involuntary
//!     context switches.  74 % of the worst commit is failed attempts, on a
//!     host whose own floor under with_pmqos is 219 ns.
//!     Do not pair a whole-commit bundle total against the retry phase — it
//!     also contains the snapshot's and the successful commit's bundles (45 %
//!     of all bundling in that run) and reads over 100 %, which is how this was
//!     first got wrong.  Note also the asymmetry the whole-commit line exposes:
//!     ~7.5 us per bundle pass against ~925 ns per unbundle pass, 8x, with
//!     bundle passes the rarer of the two (0.65 vs 1.10 per slow commit).
//!   * **Why privilege never rescues it — counted, not read.**  The obvious
//!     objection to the above is that the STM has a completion guarantee for
//!     exactly this: a transaction that keeps losing claims privilege and
//!     everyone else defers to it.  The claim gate is
//!     `if(_ll_saw && !snap.m_registered_privileged)` with NO priority term, so
//!     HIGHEST is neither privileged nor excluded there — it claims on the same
//!     terms as anyone.  Yet `priv strips (Rule 0)` is 0 in every run of this
//!     harness.  Four gates stand between a losing commit and that claim, and
//!     the ll_* counters below separate them.  Four RT-host runs of 120 s,
//!     14.3 / 13.6 / 15.9 / 8.3 M warm commits, whose MAXes span 41 us to 4 ms and
//!     which agree on these shares to a few points regardless — which is what
//!     makes the shares a property of the workload and not of the schedule:
//!       - the retry threshold, `my_tx_retries >= clamp(sig_C*2, 3,
//!         hardware_concurrency())` — **the largest blocker by far, 57-66 % of
//!         ticks**.
//!       - the per-linkage window reset, 25-34 % — `LivelockProbe::state()`
//!         holds ONE `linkage_id`, and a multi-nodal commit negotiates on
//!         several, so each switch discards the accumulated window.
//!       - `tags_owned == tags_total`, 9-11 % — the condition a displaced
//!         thread necessarily fails, i.e. the one that most needs the rescue.
//!       - `m_tagged_linkages.empty()`, the gate OUTSIDE the probe: 7-9 %
//!         (entries 3.08 vs ticks 2.83 in the cleanest run).  On a shared
//!         container this one dominates instead — entries 0.63 per slow
//!         commit, ~37 % never negotiating at all, because a plain CAS loss is
//!         not a negotiation.  It is the one figure of the four that is a
//!         property of the host rather than of the STM.
//!     The four are exhaustive and mutually exclusive to the last digit in
//!     every run: 0.69 + 0.28 + 1.86 = 2.83 in the cleanest one, 0.72 + 0.27 +
//!     1.88 + 0.0017 = 2.87 in the one where privilege fired.
//!     REGIME — two of those three runs were NOT the shipped one, and how they
//!     failed to be is the trap this harness now guards.  They were `taskset`
//!     to the isolated cores with an outer `chrt -f 20`, and the chrt did
//!     NOTHING: every thread body opens with os_be_ordinary(), correctly (see
//!     the comment there — PTHREAD_INHERIT_SCHED), so the elevation was
//!     demoted at thread start and all four threads ran SCHED_OTHER sharing
//!     two cores.  The only trace was the missing `OS arm:` line.  Both then
//!     hit a 4.006 ms MAX, reproducible to 0.5 us across runs and landing in a
//!     DIFFERENT phase each time — a quantised external stall, not the STM.
//!     The harness now adopts an inherited RT priority and prints that it did,
//!     and warns when FIFO is on without per-thread pinning; a `taskset` mask
//!     is not pinning.  Use KAME_MIX_OS_FIFO / KAME_MIX_OS_PIN, not chrt.
//!     One more trap in the same family: `isolcpus=2,3` REMOVES those CPUs
//!     from the default affinity mask, so a run without an explicit taskset
//!     sees only the housekeeping cores — the shipped-shape run above reports
//!     `acq->cpu1, others->cpu0, 2 CPUs` and got its 45 us there, on the NOISY
//!     pair.  To land on the isolated cores, taskset INTO them AND let the
//!     harness pin within that mask.
//!     WHETHER THE THRESHOLD CAN BE MET AT ALL depends on the host, and the
//!     first answer here was a container artifact stated as a general one.
//!     Do NOT derive it from the outer attempt count either — "attempts peak
//!     at 3 so retries peak at 2" ignores Node::snapshot()'s own retry loop,
//!     which increments the same `m_tx_retry_count` live and only restores it
//!     when GuardSnapshotRetryCount leaves scope, so a tick taken inside it
//!     sees more than any attempt count predicts.  Measured (the `retry
//!     margin` line, which prints REACHABLE/UNREACHABLE outright, PER RUN,
//!     because it is not a constant):
//!       container                max 2 vs 4  UNREACHABLE   0 verdicts
//!       isolated pair, invol 0   max 3 vs 4  UNREACHABLE   0 verdicts
//!       housekeeping, invol 137  max 4 vs 4  REACHABLE     3 verdicts
//!     The workload sits EXACTLY on the boundary and which side it lands on is
//!     set by how disturbed the host is: privilege fires when the host
//!     disturbs the thread and not otherwise.  Defensible as a design — the
//!     probe is for pathological cases — but it does mean privilege is not
//!     what bounds the CLEAN-host tail.  Mean my_tx_retries at a tick is 0.12
//!     in all three: the probe is nearly always looking at a FIRST attempt.
//!     "Peaks at 2, therefore unreachable" was published here once off the
//!     container alone and refuted by the next run.  Do not generalise a
//!     contention level; the harness reprints the verdict every run for that
//!     reason.
//!     What is NOT broken: the gate itself, on any host — every verdict ever
//!     reached converted, 4 for 4 across all runs.  So HIGHEST's tail is
//!     not a privilege FAILURE but a privilege RARITY: the probe is calibrated
//!     for sustained mutual livelock at ~1 firing per 4.5 M commits, and a
//!     3-attempt CAS race that resolves is not what it is looking for.
//!     Lowering the threshold is the lever if one is ever wanted; this file
//!     establishes only that it is the binding one, not that it should move.
//!     Three things that could have explained that tail instead, all checked
//!     on the RT host in the same clean configuration and all negative, so a
//!     future reader does not re-open them:
//!       - **Mapping (precondition 4).**  `rt_violations` = 0 and the pool's
//!         mapped bytes flat at 32 MiB across the window, with or without
//!         KAME_MIX_RESERVE_MIB.  Worth having checked rather than assumed —
//!         prewarm provisions size classes, not address space, and with
//!         cross-thread frees the two come apart — but at this working-set
//!         size (16 nodes, five payload clones per commit) the in-flight set
//!         never approaches one region.
//!       - **Warm-up residue.**  Slow commits run 0.68x uniform in the first
//!         warm second and the MAX landed 20.7 s in.  500 ms is enough.  (Two
//!         earlier runs had put the MAX at 4.5 s and 4.6 s, which looked like a
//!         fixed early event; three more put it at 3.0, 9.1 and 20.7 s.  A MAX
//!         is one sample.)
//!       - **The machine.**  latency_floor under with_pmqos on the same core:
//!         MAX 219 ns, nothing over a microsecond in 370 M samples.
//!   * Refuted: the cross-subtree role is NOT what the 12-13 ms residue is made
//!     of.  Unpinned at a 1 ms budget, turning it off halves the clipped
//!     population (0.077 % -> 0.039 %) and leaves MAX at 12.0 -> 13.0 ms.
//!     Narrowing XSecondaryDriver's scope is worth doing for throughput; it
//!     does not buy the tail.
//!   * Starving that same NORMAL peer did NOT pin the acquisition thread —
//!     acquisition sped back up, because a contender that is not running is
//!     not contending.  The never-expiring-privilege pin was NOT reproduced
//!     this way, and probably cannot be: privilege claims are probe-gated, so
//!     a starved thread is overwhelmingly likely to be starved while holding
//!     nothing.  `transaction_priv_expiry_test` stays the deterministic
//!     instrument for that; this arm reaches the deployment *shape*, not that
//!     specific interleaving.
//!   * `KAME_MIX_OS_STARVE=2` with the default two spinners drives the
//!     housekeeping threads to **zero** commits and trips the stall detector.
//!     Read that as an over-harsh configuration, not as an STM livelock: a
//!     count of 0 means the thread never completed even one transaction, i.e.
//!     it never got the CPU, which is what three SCHED_IDLE threads sharing
//!     one core with two spinners buys.  Use `KAME_MIX_OS_LOAD=1` or
//!     `KAME_MIX_OS_STARVE=1` to keep the holder schedulable enough to be
//!     interesting.
//!
//! Livelock is detected as STALL: a per-thread commit counter that stops
//! advancing for KAME_MIX_STALL_SECS (default 5) while wall time advances.
//! No starvation handler is installed (kamestm default), so a livelock
//! manifests as no-progress rather than an exception — which is what the
//! field sees, since a stuck UI iterate_commit never returns to the event
//! loop.
//!
//! Knobs (env):
//!   KAME_MIX_SECS            run length, default 10 (ctest); set 60+ to soak
//!   KAME_MIX_STALL_SECS      stall threshold, default 5
//!   KAME_MIX_HIGHEST_DUTY_US pause between records, default 0 = flat out
//!   KAME_MIX_UI_PERIOD_US    pause between UI actions, default 0 = flat out
//!   KAME_MIX_NORMALS         extra NORMAL driver threads, default 1
//!   KAME_MIX_SCRIPTING       SCRIPTING threads, default 1 — the field has TWO
//!                            lowprio threads (main UI_DEFERRABLE + the Python
//!                            thread), and the bench already knows lowprio
//!                            threads starve each *other* at 2+, not at 1
//!   KAME_MIX_UI_WIDE         every Nth UI action is a root-scope Tx,
//!                            default 8; 1 = hostile (every action wide)
//!   KAME_MIX_LEAVES          leaves per subtree, default 4.  The field root
//!                            has ~10^3 nodes, so a root bundle costs ms and
//!                            the invalidation window for a wide UI Tx is
//!                            enormous — a 16-node tree cannot reproduce that.
//!                            Set 64-256 to model a real measurement tree
//!   KAME_MIX_OS_FIFO         >0 = acquisition thread at SCHED_FIFO of that
//!                            priority (Linux; skipped with a notice when not
//!                            permitted).  The rest stay SCHED_OTHER
//!   KAME_MIX_OS_PIN          1 = acquisition alone on the last CPU, everyone
//!                            else on CPU 0 — the shape isolcpus produces
//!   KAME_MIX_OS_STARVE       1 = UI+SCRIPTING to SCHED_IDLE, 2 = NORMAL too
//!   KAME_MIX_OS_LOAD         SCHED_OTHER spinners on the housekeeping CPU at
//!                            starve level 2, default 2 (see the caveat below)
//!   KAME_MIX_NORMAL_XSUBTREE 1 = every 4th NORMAL Tx spans the acquiring
//!                            driver's subtree (XSecondaryDriver's shape)
//!   KAME_MIX_ACQ_NORMAL      1 = acquisition thread runs at NORMAL instead of
//!                            HIGHEST: the control arm that attributes any
//!                            stall to the HIGHEST-ification or acquits it.
//!                            **This is the arm KAME actually ships** —
//!                            XPrimaryDriverWithThread::AcquisitionPriority is
//!                            ScopedPriority(NORMAL) plus an OS elevation; the
//!                            kamestm HIGHEST tier was retired for KAME because
//!                            per-record analyses cannot honour its
//!                            precondition.  So the HIGHEST arm measures the
//!                            library's ceiling and this one measures the
//!                            product.
//!   KAME_MIX_RT_POOL         how much of kamepoolalloc's realtime CONTRACT to
//!                            honour.  0 = none; 1 = kame_pool_set_realtime_
//!                            mode(1), which is what kame/main.cpp does and
//!                            therefore what the SHIPPED application is at;
//!                            2 = + KAME_RT_DEFER on the acquisition thread
//!                            and kame_pool_rt_drain() every
//!                            KAME_MIX_RT_DRAIN_EVERY records (precondition 5);
//!                            **3 = KAME_RT_STRICT instead of DEFER, and the
//!                            DEFAULT.**  Use 0 to reproduce anything measured
//!                            before 2026-08, and 1 when the question is what
//!                            KAME ships rather than what the STM can do.
//!                            Why a LATENCY harness defaults to it: STRICT is
//!                            the only level that drops this thread's
//!                            CROSS-THREAD dealloc batch to per-free flushing,
//!                            so no single free inherits the batch's CAP=1024
//!                            sort+merge+CAS.  The acquisition thread frees
//!                            cross-thread precisely when a peer allocates on
//!                            ITS subtree — whoever drops the last reference
//!                            frees the payload — i.e. precisely under
//!                            KAME_MIX_NORMAL_XSUBTREE.  Measured on the
//!                            PREEMPT_RT host, FIFO + isolation, 60 s arms,
//!                            slow (>= 50 us) commits per million:
//!
//!                              level 0 (none)        28.8   MAX 92,633 ns
//!                              level 1 (mode)        31.0   MAX 101,584
//!                              level 2 (DEFER)       29.3   MAX  94,161
//!                              level 3 (STRICT)       8.3   MAX  66,737
//!
//!                            3.5x, and only at STRICT — mode and DEFER are
//!                            within noise.
//!                            RETRACTED, and worth keeping visible because the
//!                            error is the easy one to repeat: this used to
//!                            read "MAX 66,737 ns is BELOW the 67,879 ns
//!                            latency_floor measures for this host, so level 3
//!                            is no longer distinguishable from the machine."
//!                            Those arms ran WITH isolation (see the line
//!                            above), and 67,879 ns is latency_floor's
//!                            *un-isolated* row.  The right floor for this
//!                            regime is 17,030 ns, so 66,737 is 3.9x the
//!                            machine, not level with it — the ~50 us above
//!                            the floor is the STM's own, which is exactly
//!                            what the phase split later measured directly
//!                            (46,670 of 50,707 ns in the retry path).  The
//!                            check that would have caught it at the time:
//!                            removing the floor entirely (with_pmqos, 219 ns)
//!                            moved MAX only 66.7 -> 53.1 us.  A tail that was
//!                            "the machine" cannot survive deleting the
//!                            machine.  Quote a floor from the SAME
//!                            configuration row, always.
//!                            Throughput
//!                            went UP 8 % (50.7k -> 54.7k), so the contract's
//!                            documented "STRICT costs ~47 % of cross-thread
//!                            small-free throughput" does not reach this
//!                            workload.
//!                            CONSEQUENCE FOR THE NUMBERS BELOW: everything
//!                            recorded in this header predates the default and
//!                            was measured at level 0.  Comparisons across
//!                            that boundary are not valid.
//!   KAME_MIX_WB_US           the ScopedWaitBudget over the record commit, µs.
//!                            Default 20000, mirroring
//!                            XPrimaryDriver::downstreamWaitBudgetUS().  0
//!                            removes it.  Sweepable because at NORMAL it is
//!                            the ONLY thing bounding the record commit —
//!                            HIGHEST leaves the negotiator's round loop before
//!                            it can sleep, NORMAL does not — so the measured
//!                            MAX pins to this value, and what a deployment can
//!                            actually buy is read off the throughput it costs
//!                            to lower it.  (Not a hard cap: the wait behind a
//!                            live privileged peer stays budget-exempt.)

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include "latency_hist.h"
#ifndef DISABLE_POOL_ALLOCATOR
#  include "kame_pool.h"
#endif
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <cstring>
#if defined(__linux__)
#  include <sys/resource.h>   // getrusage(RUSAGE_THREAD)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/prctl.h>
#endif

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        long m_x = 0;
    };
};
typedef Transactional::Transaction<MyNode> Tr;
typedef Transactional::Snapshot<MyNode> Ss;

#if KAME_STM_NEG_DIAG
//! bundle() + unbundle() wall time so far on this thread.  Read as a running
//! total and DIFFERENCED by the caller — the counters are cumulative per
//! commit, so only a difference across a phase boundary is attributable to
//! that phase.  \sa PhaseStat::rbu
static inline std::uint64_t bundle_unbundle_ns() {
    const auto &d = Transactional::detail::neg_diag();
    return d.bundle_ns + d.unbundle_ns;
}
#endif
static long env_long(const char *name, long defv) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atol(v) : defv;
}

// ---------------------------------------------------------------- OS class
//! The dimension this test did not have.  Every thread here has always been
//! SCHED_OTHER, so CFS runs all of them regularly and the field's rare
//! livelock stays rare: whoever holds a privilege is always scheduled soon
//! enough to finish it.
//!
//! That is not what an RT deployment looks like.  `AcquisitionPriority` keeps
//! only the OS elevation now that STM-HIGHEST is retired, so on such a host
//! the acquisition thread is `SCHED_FIFO` on an isolated core while the UI and
//! scripting threads share a loaded housekeeping core.  And the STM
//! deliberately does not rescue that: NORMAL privilege never expires (it *is*
//! the completion guarantee, and the TLA+ liveness argument assumes it
//! persists until its holder finishes), and the wait behind a live privilege
//! is exempt from the wait budget.  The bound is therefore the holder's
//! scheduling delay and nothing else — which makes it a *configuration*
//! property of the deployment, not a property of the STM.
//!
//! These knobs make that configuration reachable so the consequence can be
//! observed rather than argued about.  A stall is still the verdict.
//!
//! Nothing here ever puts a spinning thread on SCHED_FIFO: equal-priority FIFO
//! threads do not preempt one another, and the load arm would wedge the box
//! rather than starve a holder.  Starvation is modelled with SCHED_IDLE, which
//! cannot.
static bool os_set_policy(int policy, int prio) noexcept {
#if defined(__linux__)
    sched_param sp;
    std::memset( &sp, 0, sizeof(sp));
    sp.sched_priority = prio;
    return pthread_setschedparam(pthread_self(), policy, &sp) == 0;
#else
    (void)policy; (void)prio;
    return false;
#endif
}
//! Explicit, at the top of every thread body.  pthread_create defaults to
//! PTHREAD_INHERIT_SCHED, so a thread spawned after another has elevated
//! itself would silently come up elevated too — the bug this project has
//! already paid for once, in bench_rt_wcet.
static void os_be_ordinary() noexcept {
#if defined(__linux__)
    os_set_policy(SCHED_OTHER, 0);
#endif
}
static void os_be_starved() noexcept {
#if defined(__linux__)
    if( !os_set_policy(SCHED_IDLE, 0)) os_be_ordinary();
#endif
}
static void os_pin(long cpu) noexcept {
#if defined(__linux__)
    if(cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO( &set);
    CPU_SET((int)cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}
static void pause_us(long us) {
    if(us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(us));
}

int main() {
    const long secs        = env_long("KAME_MIX_SECS", 10);
    const long stall_secs  = env_long("KAME_MIX_STALL_SECS", 5);
    const long hi_duty_us  = env_long("KAME_MIX_HIGHEST_DUTY_US", 0);
    const long ui_period_us= env_long("KAME_MIX_UI_PERIOD_US", 0);
    const long n_normals   = env_long("KAME_MIX_NORMALS", 1);
    const long n_scripting = env_long("KAME_MIX_SCRIPTING", 1);
    const long ui_wide     = env_long("KAME_MIX_UI_WIDE", 8);
    const long n_leaves    = env_long("KAME_MIX_LEAVES", 4);
    const bool acq_normal  = env_long("KAME_MIX_ACQ_NORMAL", 0) != 0;
    //! \sa the KAME_MIX_WB_US block in the header comment.  Default mirrors
    //! XPrimaryDriver::downstreamWaitBudgetUS(); 0 removes the budget.
    const long wb_us       = env_long("KAME_MIX_WB_US", 20000);
    const long os_fifo_req = env_long("KAME_MIX_OS_FIFO", 0);
    //! 0 = off, 1 = the lowprio tiers (UI + SCRIPTING) go SCHED_IDLE,
    //! 2 = the NORMAL peers as well.  The two levels ask different questions.
    //! Expiry is a lowprio-only mechanism, so level 1 starves holders whose
    //! privilege the STM *can* revoke and therefore tests that rescue path.
    //! Level 2 starves a NORMAL holder, whose privilege never expires and
    //! behind which the wait budget is exempt — there the only bound left is
    //! the OS scheduler, which is the case worth quantifying before choosing
    //! a policy for raiseAcquisitionOSPriority_().
    const long os_starve   = env_long("KAME_MIX_OS_STARVE", 0);
    //! 1 = every 4th NORMAL transaction spans the acquiring driver's subtree
    //! (the XSecondaryDriver / ms-analysis role).  Off by default so the
    //! existing arms are unchanged.
    const bool normal_xsub = env_long("KAME_MIX_NORMAL_XSUBTREE", 0) != 0;
    //! >0 turns the record-commit distribution into an assertion: any record
    //! commit longer than this many microseconds fails the run.  0 (default)
    //! reports the distribution and asserts nothing, because an absolute
    //! latency is a property of the host, not of the STM — the same split
    //! bench_rt_wcet makes between its machine-independent violation count and
    //! its machine-specific histogram.
    const long deadline_us = env_long("KAME_MIX_DEADLINE_US", 0);
    //! >0 arms a breaktrace: the FIRST record commit longer than this many
    //! microseconds writes a marker into ftrace and switches tracing off, so
    //! the buffer freezes holding whatever ran just before it.  The same
    //! instrument cyclictest's --breaktrace is, aimed at a commit instead of a
    //! timer wake-up — because hunting a rare fixed-cost event by reading a
    //! running trace is hopeless, while catching it in the act is routine.
    //! Needs tracefs mounted and root; says so and stays disarmed otherwise.
    const long trace_us = env_long("KAME_MIX_TRACE_US", 0);
    //! Samples in the first this-many milliseconds are counted but kept OUT of
    //! the histogram and cannot fire the breaktrace.  Without it the largest
    //! sample of every run is a cold-start artefact and the breaktrace can
    //! never catch anything else — the 2026-08 RT investigation had to disable
    //! the allocator's pre-fill entirely to get past it.
    const long warmup_ms = env_long("KAME_MIX_WARMUP_MS", 500);
    //! Precondition 2 of the realtime contract: prewarm from the realtime
    //! thread, before the time-critical section.  On by default because the
    //! contract requires it and because this test previously did not do it —
    //! which is exactly how it came to measure a 400 us first-commit spike
    //! (the pool's per-slot next-pointer pre-fill faulting 5 size classes x 64
    //! pages) and mistake it for a recurring event.  Set 0 to reproduce that.
    const bool do_prewarm  = env_long("KAME_MIX_PREWARM", 1) != 0;
    //! How much of the pool's realtime contract to honour.  \sa the header.
    //! **Default 3 (STRICT) since the 2026-08 measurement below**: this is a
    //! realtime harness, and running it with the contract unhonoured measured
    //! the allocator's ungated free path rather than the STM.  Set 0 to
    //! reproduce the pre-2026-08 numbers, or 1 for shipped-KAME fidelity —
    //! `kame/main.cpp` sets the process-wide mode and nothing marks the
    //! acquisition thread, so the application is at level 1, not 3.
    const long rt_pool     = env_long("KAME_MIX_RT_POOL", 3);
    //! Drain every N records, not every record.  Precondition 5 says "from a
    //! non-critical phase — between control cycles"; this harness has no
    //! trough (records come at ~10^5/s, a real driver at 1..10^4 Hz), so
    //! draining per record costs 8x throughput and models nothing.  Separated
    //! from the level so DEFER/STRICT can be A/B'd without it.  0 = never.
    const long rt_drain_every = env_long("KAME_MIX_RT_DRAIN_EVERY", 1024);
    //! Pre-map this many MiB of pool regions up front (32 MiB granularity,
    //! prefaulted).  Precondition 4 of the realtime contract is that the
    //! working set does not GROW during the section, and prewarm does not
    //! give you that: it provisions size-class capacity, not mapped address
    //! space.  With cross-thread frees the two come apart — a block freed by
    //! a peer does not land back on this thread's freelist, so a steady-state
    //! in-flight set larger than what prewarm covers keeps claiming chunks,
    //! then regions, and a region is an mmap on the measured path.  0 = off,
    //! which is how everything so far was measured.
    const long reserve_mib = env_long("KAME_MIX_RESERVE_MIB", 0);
    //! Split the timed region into snapshot / payload-write / commit by
    //! expanding iterate_commit by hand — its body is exactly
    //! `for(Transaction tr(node);; ++tr) { closure(tr); if(tr.commit()) ... }`
    //! so the three phases are the ctor+`++tr`, the closure, and commit().
    //! The last instrument left: every retry loop is counted and quiet, no
    //! syscall, no fault, no sleep, and the machine floor is 219 ns, so the
    //! tail is one slow PASS and only a timer can say which part of it.
    //! Costs four now_ns() per attempt (~72 ns on an 814 ns commit), which
    //! moves the median a few per cent and cannot hide 50 us.  Off by
    //! default; the default path stays the real iterate_commit.
    const bool phase_mode = env_long("KAME_MIX_PHASE", 0) != 0;
    //! Per-thread timer slack for the acquisition thread, in ns.  Linux
    //! defaults to 50 us and applies it to every `futex` timeout a
    //! SCHED_OTHER task arms — which is the negotiator's CV chunk, i.e. it
    //! lands directly on the record commit's tail.  (RT scheduling classes
    //! get zero slack, which is one reason a FIFO arm and a SCHED_OTHER arm
    //! are not the same measurement.)  0 = leave the default alone.
    const long timerslack  = env_long("KAME_MIX_TIMERSLACK_NS", 0);
    const bool os_pin_on   = env_long("KAME_MIX_OS_PIN", 0) != 0;
    std::printf("mixed-priority livelock hunt: %lds, stall>%lds fails, "
                "acq=%s duty %ldus, UI period %ldus, +%ld NORMAL, "
                "+%ld SCRIPTING, %ld leaves/subtree, record wait budget ",
                secs, stall_secs,
                acq_normal ? "NORMAL(SHIPPED)" : "HIGHEST(library ceiling)",
                hi_duty_us, ui_period_us, n_normals, n_scripting, n_leaves);
    //! Printed on the banner because at NORMAL the MAX below pins to it, so a
    //! run's headline number is unreadable without knowing which budget
    //! produced it.
    if(wb_us) std::printf("%ldus\n", wb_us);
    else      std::printf("NONE\n");

    // Probe the OS arm up front so an unprivileged run says so instead of
    // reporting a green RT result it never ran.
    //! Derive the CPUs from this process's AFFINITY MASK, not from the online
    //! count: `taskset -c 0,1 ./test` then chooses which cores the arm uses,
    //! which is the discriminator for "is the penalty about crossing cores, or
    //! about crossing onto a nohz_full one" — no extra knob needed.
    std::vector<long> cpus;
#if defined(__linux__)
    {
        cpu_set_t set;
        CPU_ZERO( &set);
        if(sched_getaffinity(0, sizeof(set), &set) == 0) {
            for(int c = 0; c < CPU_SETSIZE; ++c)
                if(CPU_ISSET(c, &set)) cpus.push_back(c);
        }
    }
#endif
    if(cpus.empty()) cpus.push_back(0);
    const long ncpu = (long)cpus.size();
    //! An outer `chrt -f N ./this_test` used to be silently thrown away, and it
    //! cost a whole measurement session before anyone noticed.  Every thread
    //! body opens with os_be_ordinary() — correctly, because
    //! PTHREAD_INHERIT_SCHED would otherwise leak one thread's elevation into
    //! the next — so the inherited RT class is demoted at the top of each
    //! thread and the run is SCHED_OTHER throughout, while the command line
    //! and the operator both say SCHED_FIFO.  The only visible trace was the
    //! absence of the `OS arm:` line below, which is not something a reader
    //! notices.  Adopt the inherited priority instead: `chrt -f 20 ./test` and
    //! `KAME_MIX_OS_FIFO=20 ./test` are the same request.  Setting the variable
    //! explicitly always wins, including to 0, which opts out.
    long os_fifo_eff = os_fifo_req;
#if defined(__linux__)
    {   const int pol = ::sched_getscheduler(0);
        struct sched_param sp{};
        if((pol == SCHED_FIFO || pol == SCHED_RR)
                && (::sched_getparam(0, &sp) == 0) && (sp.sched_priority > 0)) {
            if( !std::getenv("KAME_MIX_OS_FIFO")) {
                os_fifo_eff = sp.sched_priority;
                std::printf("  NOTE: started under SCHED_%s %d (chrt?) — "
                            "adopting it as KAME_MIX_OS_FIFO=%ld.  Every thread "
                            "body resets its own policy, so without this the "
                            "run would silently have been SCHED_OTHER.\n",
                            (pol == SCHED_FIFO) ? "FIFO" : "RR",
                            sp.sched_priority, os_fifo_eff);
            }
            else if(os_fifo_req != sp.sched_priority)
                std::printf("  NOTE: started under SCHED_%s %d but "
                            "KAME_MIX_OS_FIFO=%ld is set and wins.\n",
                            (pol == SCHED_FIFO) ? "FIFO" : "RR",
                            sp.sched_priority, os_fifo_req);
        }
    }
#endif
    const long os_fifo = os_fifo_eff;   //!< what the run actually uses
    bool fifo_ok = false;
    if(os_fifo > 0) {
        fifo_ok = os_set_policy(SCHED_FIFO, (int)os_fifo);
        os_be_ordinary();               // probe only; the thread sets its own
        if( !fifo_ok)
            std::printf("  NOTE: SCHED_FIFO %ld was requested but is not "
                        "permitted (need CAP_SYS_NICE or RLIMIT_RTPRIO) — the "
                        "OS arm is SKIPPED, this run is SCHED_OTHER "
                        "throughout.\n", os_fifo);
    }
    if(os_pin_on && (ncpu < 2)) {
        std::printf("  NOTE: KAME_MIX_OS_PIN needs >= 2 usable CPUs (have "
                    "%ld) — pinning SKIPPED.\n", ncpu);
    }
    const bool pin_ok = os_pin_on && (ncpu >= 2);
    //! Acquisition alone on the LAST allowed CPU, everyone else on the first:
    //! the shape `isolcpus` produces, without needing the kernel parameter.
    const long cpu_acq   = pin_ok ? cpus.back()  : -1;
    const long cpu_house = pin_ok ? cpus.front() : -1;
    if(os_fifo > 0 || os_starve || pin_ok)
        std::printf("  OS arm: fifo=%s starve(SCHED_IDLE lowprio)=%s "
                    "pin=%s (acq->cpu%ld, others->cpu%ld), %ld CPUs\n",
                    fifo_ok ? "yes" : "no",
                    (os_starve >= 2) ? "lowprio+NORMAL" :
                        (os_starve ? "lowprio" : "no"),
                    pin_ok ? "yes" : "no", cpu_acq, cpu_house, ncpu);
    //! The combination this project measured as catastrophic and then shipped
    //! a rule about — "FIFO and isolation ship together or neither ships" —
    //! used to run without a word: the elevated thread preempts the very CFS
    //! holders it then waits behind, contenders collapse to ~150 commits/s,
    //! and the tail goes to tens of milliseconds.  `taskset` on the command
    //! line does NOT satisfy this: it restricts the mask that every thread
    //! shares, whereas the rule needs the deadline thread ALONE on a core.
    if(fifo_ok && !pin_ok)
        std::printf("  WARNING: SCHED_FIFO is on but per-thread pinning is "
                    "OFF (KAME_MIX_OS_PIN=1).  The elevated thread will "
                    "preempt the peers it then waits behind — a measured "
                    "priority inversion, not a realtime configuration.  Any "
                    "tail from this run describes that, and taskset alone "
                    "does not fix it.\n");

#ifndef DISABLE_POOL_ALLOCATOR
    //! Process-wide half of KAME_MIX_RT_POOL, before any thread starts.
    //! Level >= 1 is what `kame/main.cpp:562` already does, so anything below
    //! it models the pool MORE loosely than the application does.
    if(rt_pool >= 1) kame_pool_set_realtime_mode(1);
    if(reserve_mib > 0) {
        unsigned got = kame_pool_reserve_regions(
            (unsigned)((reserve_mib + 31) / 32), /*prefault=*/1);
        std::printf("  pre-reserved %u region(s) = %u MiB, prefaulted\n",
                    got, got * 32u);
    }
    std::printf("  pool realtime contract: %s\n",
        rt_pool <= 0 ? "none (preconditions 1/3/5 unmet — the pre-2026-08 "
                       "baseline; NOT the default)" :
        rt_pool == 1 ? "mode only (matches kame/main.cpp, i.e. shipped KAME)" :
        rt_pool == 2 ? "mode + RT_DEFER on acq + rt_drain in the trough" :
                       "mode + RT_STRICT on acq + rt_drain in the trough");
#endif

    // The measurement tree: root -> {devA, devB, panel}, four leaves each.
    // devA is the acquiring driver's subtree; entriesA models its scalar
    // entries, written by the demoted downstream.
    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> devA(MyNode::create<MyNode>());
    shared_ptr<MyNode> devB(MyNode::create<MyNode>());
    shared_ptr<MyNode> panel(MyNode::create<MyNode>());
    root->insert(devA); root->insert(devB); root->insert(panel);
    std::vector<shared_ptr<MyNode>> leavesA, leavesB, leavesP;
    for(long i = 0; i < n_leaves; ++i) {
        shared_ptr<MyNode> a(MyNode::create<MyNode>());
        shared_ptr<MyNode> b(MyNode::create<MyNode>());
        shared_ptr<MyNode> p(MyNode::create<MyNode>());
        leavesA.push_back(a); devA->insert(a);
        leavesB.push_back(b); devB->insert(b);
        leavesP.push_back(p); panel->insert(p);
    }
    //! The roles deliberately poke DIFFERENT leaves of the acquiring driver's
    //! subtree — 0 the demoted downstream, 1 the UI's snapshot, 2 the UI's
    //! mid-acquisition write, 3 the scripting poke — so the indices carry
    //! meaning and must not all collapse onto leaf 0.  Wrapping spreads them
    //! as far as a small tree allows; the point of the helper, though, is that
    //! they were RAW: `KAME_MIX_LEAVES` below 4 indexed out of bounds and
    //! handed the STM a garbage node, aborting with a `domain_error` out of
    //! the payload lookup — which reads as an STM bug and is a harness one.
    auto lA = [&](std::size_t i) -> const shared_ptr<MyNode> & {
        return leavesA[i % leavesA.size()];
    };

    enum {T_HIGHEST = 0, T_DOWNSTREAM = 1, T_UI = 2, T_SCRIPT0 = 3};
    const int T_NORMAL0 = T_SCRIPT0 + (int)n_scripting;
    const int nthreads = T_NORMAL0 + (int)n_normals;
    //! One cache line each.  Unpadded, all of these sit in one or two lines
    //! and every role's fetch_add invalidates the line for every other role —
    //! the harness would be generating the coherence traffic it is measuring,
    //! and the distortion grows with the number of roles, i.e. exactly with
    //! the arms being compared.  128 rather than 64: Intel's L2 adjacent-line
    //! prefetcher pulls pairs.
    struct alignas(128) Counter {
        std::atomic<uint64_t> v{0};
    };
    static_assert(sizeof(Counter) == 128, "Counter must occupy a whole pair");
    std::vector<Counter> progress_(nthreads);
    //! Thin accessor so the (many) call sites keep reading `progress[t]`.
    struct ProgressView {
        Counter *p;
        std::atomic<uint64_t> &operator[](size_t i) const { return p[i].v; }
    } progress{progress_.data()};
    for(auto &c : progress_) c.v.store(0);
    std::atomic<bool> stop{false};
    std::vector<std::thread> ts;
    //! Written only by the acquisition thread, read only after join().
    Hist acq_hist;
    acq_hist.reset();
    std::atomic<uint64_t> cold_n{0};   //!< commits dropped as warm-up
    //! Written only by the acquisition thread, read after its join.
    std::uint64_t acq_max_seen = 0, acq_max_at = 0, acq_slow_1st_sec = 0;
    //! Sampled when the warm window closes, so start-up mapping is excluded
    //! and what is reported is growth DURING the measurement.
    unsigned long long rt_viol_warm = 0, rt_reclaim_warm = 0, rt_unmap_warm = 0;
    std::size_t rt_bytes_warm = 0;
    bool rt_warm_sampled = false, rt_warm_first = true;
    //! KAME_MIX_PHASE aggregate: sums over the SLOW population and the worst
    //! commit's own triple, which is the one that has to add up.
    //! `rbu` is bundle+unbundle wall time accumulated INSIDE the retry segment
    //! only — the failed commitOrNext() calls, differenced across the same
    //! boundaries as `retry` itself.  A per-commit bundle total cannot answer
    //! "how much of the RETRY is bundling", because it also contains the
    //! bundles of the snapshot and of the successful commit; paired against
    //! the retry phase it can and did exceed 100 %.
    struct PhaseStat {
        std::uint64_t n = 0, snap = 0, write = 0, commit = 0, retry = 0,
                      rbu = 0;
        std::uint64_t max_dt = 0, max_snap = 0, max_write = 0,
                      max_commit = 0, max_retry = 0, max_rbu = 0;
        void add(std::uint64_t dt, std::uint64_t s, std::uint64_t w,
                 std::uint64_t c, std::uint64_t r, std::uint64_t bu) {
            ++n; snap += s; write += w; commit += c; retry += r; rbu += bu;
            if(dt > max_dt) { max_dt = dt; max_snap = s;
                              max_write = w; max_commit = c;
                              max_retry = r; max_rbu = bu; }
        }
    } phase_slow;
    //! The acquisition thread's own kernel entries over the measured window.
    //! Syscalls on this path are argued away easily — HIGHEST never reaches
    //! negotiate_sleep's futex or the fair-spin's sched_yield, the pool makes
    //! none on the free path under RT_STRICT, and now_ns() is vDSO — but a
    //! PAGE FAULT is a kernel entry that is nobody's syscall and that nothing
    //! here counted.  Two getrusage(RUSAGE_THREAD) calls, both outside the
    //! timed region, settle it instead of arguing it.
    long ru_min_warm = 0, ru_maj_warm = 0, ru_nvcsw_warm = 0, ru_nivcsw_warm = 0;
    long ru_min_end = 0, ru_maj_end = 0, ru_nvcsw_end = 0, ru_nivcsw_end = 0;
    //! Retry accounting for the slow tail.  Written only by the acquisition
    //! thread; `sysd` comes from the other roles' progress counters, so a slow
    //! commit reports whether the rest of the tree kept committing while it
    //! was stuck.
    Retries acq_retries;
    //! Threshold for "slow", in ns.  Default 50 us: comfortably above this
    //! host class's OS floor (rtla osnoise put it at 17 us) and below the
    //! ~90 us residue the RT investigation left unexplained.
    const long slow_ns = env_long("KAME_MIX_SLOW_NS", 50000);
#if KAME_STM_NEG_DIAG
    //! Where a slow commit's time actually went, from inside the negotiator.
    //! The measurement this test could not make until now: MAX = budget +
    //! ~200 us was reproducible but unattributed, and "the budget-exempt wait
    //! behind a live privileged peer" was a hypothesis with the right shape
    //! and no evidence.  `rounds_exempt` / `slept_exempt_ns` are that
    //! evidence, because the exemption is the ONLY route by which a wait can
    //! outlive the budget: every other sleep site clamps its chunk to the
    //! remaining budget and every round re-checks it at the top.
    struct SlowDiag {
        std::uint64_t n = 0, rounds = 0, rounds_exempt = 0, sleeps = 0,
                      slept_ns = 0, slept_exempt_ns = 0, req_ns = 0,
                      spins = 0, spin_ns = 0, entries = 0, sleeps_priv = 0,
                      late_max_ns = 0, tail_spins = 0, tail_spin_ns = 0,
                      commit_cas = 0, bundle_cas = 0, snap_cas = 0,
                      ll_ticks = 0, ll_resets = 0, ll_no_tags = 0,
                      ll_few_retries = 0, ll_verdicts = 0,
                      priv_tries = 0, priv_grants = 0,
                      ll_retry_max = 0, ll_retry_sum = 0, ll_thresh_max = 0,
                      bundle_ns = 0, bundle_calls = 0, bundle_all = 0,
                      unbundle_ns = 0, unbundle_calls = 0, unbundle_all = 0;
        //! …and the single worst commit of the run, kept whole: a mean over
        //! the slow population cannot say whether the MAX was one long exempt
        //! sleep or a hundred short budgeted ones.
        std::uint64_t max_dt = 0;
        Transactional::detail::NegDiag max_d{};
        void add(std::uint64_t dt, const Transactional::detail::NegDiag &d) {
            ++n; rounds += d.rounds; rounds_exempt += d.rounds_exempt;
            sleeps += d.sleeps; slept_ns += d.slept_ns;
            slept_exempt_ns += d.slept_exempt_ns; req_ns += d.req_ns;
            spins += d.spins; spin_ns += d.spin_ns; entries += d.entries;
            sleeps_priv += d.sleeps_priv;
            if(d.late_max_ns > late_max_ns) late_max_ns = d.late_max_ns;
            tail_spins += d.tail_spins; tail_spin_ns += d.tail_spin_ns;
            commit_cas += d.commit_cas_retries;
            bundle_cas += d.bundle_cas_retries;
            snap_cas   += d.snapshot_retries;
            ll_ticks += d.ll_ticks; ll_resets += d.ll_resets;
            ll_no_tags += d.ll_no_tags; ll_few_retries += d.ll_few_retries;
            ll_verdicts += d.ll_verdicts;
            priv_tries += d.priv_tries; priv_grants += d.priv_grants;
            ll_retry_sum += d.ll_retry_sum;
            if(d.ll_retry_max > ll_retry_max)   ll_retry_max = d.ll_retry_max;
            if(d.ll_thresh_max > ll_thresh_max) ll_thresh_max = d.ll_thresh_max;
            bundle_ns += d.bundle_ns; bundle_calls += d.bundle_calls;
            bundle_all += d.bundle_calls_all;
            unbundle_ns += d.unbundle_ns; unbundle_calls += d.unbundle_calls;
            unbundle_all += d.unbundle_calls_all;
            if(dt > max_dt) { max_dt = dt; max_d = d; }
        }
    } slow_diag;
    //! The same probe chain over EVERY warm commit, not only the slow ones.
    //! Needed because the two zero-cases are indistinguishable in the slow
    //! population alone: a probe that never ticks and a probe that ticks and
    //! never reaches a verdict both report `ll_verdicts = 0` there.  Cheap —
    //! the snapshot already happens after the timed region closes.
    struct LLAll {
        std::uint64_t n = 0, ticks = 0, resets = 0, no_tags = 0,
                      few_retries = 0, verdicts = 0, tries = 0, grants = 0,
                      retry_max = 0, retry_sum = 0, thresh_max = 0;
        void add(const Transactional::detail::NegDiag &d) {
            ++n; ticks += d.ll_ticks; resets += d.ll_resets;
            no_tags += d.ll_no_tags; few_retries += d.ll_few_retries;
            verdicts += d.ll_verdicts;
            tries += d.priv_tries; grants += d.priv_grants;
            retry_sum += d.ll_retry_sum;
            if(d.ll_retry_max > retry_max)   retry_max = d.ll_retry_max;
            if(d.ll_thresh_max > thresh_max) thresh_max = d.ll_thresh_max;
        }
    } ll_all;
#endif

    // Breaktrace plumbing.  Both descriptors are opened up front so the hot
    // path only ever does two write()s, and only once.
    int trace_marker_fd = -1, tracing_on_fd = -1;
    std::atomic<bool> trace_fired{false};
#if defined(__linux__)
    if(trace_us > 0) {
        static const char *kRoots[] = {"/sys/kernel/tracing",
                                       "/sys/kernel/debug/tracing"};
        for(const char *r : kRoots) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s/trace_marker", r);
            trace_marker_fd = ::open(buf, O_WRONLY | O_CLOEXEC);
            if(trace_marker_fd < 0) continue;
            std::snprintf(buf, sizeof(buf), "%s/tracing_on", r);
            tracing_on_fd = ::open(buf, O_WRONLY | O_CLOEXEC);
            if(tracing_on_fd >= 0) {
                std::printf("  breaktrace armed at %ld us via %s\n",
                            trace_us, r);
                break;
            }
            ::close(trace_marker_fd);
            trace_marker_fd = -1;
        }
        if(tracing_on_fd < 0)
            std::printf("  NOTE: KAME_MIX_TRACE_US needs tracefs and root — "
                        "breaktrace DISARMED (mount it and re-run as root; "
                        "the latency histogram below is unaffected).\n");
    }
#endif

    // --- The acquisition thread, oscillating exactly like finishWritingRaw:
    // record commit at HIGHEST, then the demoted downstream at NORMAL under
    // the 20 ms budget, every cycle.
    ts.emplace_back([&]{
        if(fifo_ok) os_set_policy(SCHED_FIFO, (int)os_fifo);
        else        os_be_ordinary();
        os_pin(cpu_acq);
#if defined(__linux__)
        if(timerslack > 0) ::prctl(PR_SET_TIMERSLACK, (unsigned long)timerslack);
#endif
#ifndef DISABLE_POOL_ALLOCATOR
        if(do_prewarm) {
            //! Cover the small classes the STM's Payload clones land in.
            //! Over-covering is free; missing one puts its first chunk claim
            //! back on the measured path.
            static const std::size_t kSizes[] =
                {16, 32, 48, 64, 96, 128, 192, 256, 512, 1024};
            unsigned counts[sizeof(kSizes) / sizeof(kSizes[0])];
            for(auto &c : counts) c = 64u;
            if(kame_pool_prewarm(kSizes, counts,
                                 (unsigned)(sizeof(kSizes) / sizeof(kSizes[0]))))
                std::printf("  NOTE: kame_pool_prewarm did not fit — the first "
                            "commits will show cold-path outliers.\n");
        }
        //! \sa the KAME_MIX_RT_POOL block in the header comment.  Level 2/3
        //! are per-THREAD and therefore have to be set here, on the thread
        //! that owns the deadline, not in main().
        if(rt_pool >= 2)
            kame_pool_set_realtime_thread(rt_pool >= 3 ? KAME_RT_STRICT
                                                       : KAME_RT_DEFER);
#endif
        std::uint64_t acq_iter = 0;   //!< for the periodic rt_drain below
        const std::uint64_t t_warm_end = now_ns() +
            (std::uint64_t)warmup_ms * 1000000ull;
        Transactional::ScopedPriority pr(acq_normal
            ? Transactional::Priority::NORMAL
            : Transactional::Priority::HIGHEST);
        while( !stop.load(std::memory_order_relaxed)) {
            {   // the record commit (multi-nodal, driver scope).
                //! Timed, because "the acquisition thread kept up on average"
                //! and "no record took longer than X" are different claims and
                //! only the second one is a realtime one.  The counters below
                //! answer the first; this histogram answers the second.
                //! At HIGHEST this is inert — the round loop breaks out before
                //! it can sleep — and at NORMAL it is the only bound the record
                //! commit has.  Constructed unconditionally at the default so
                //! the shape matches finishWritingRaw; wb_us == 0 opts out, to
                //! show what the budget is worth.
                std::unique_ptr<Transactional::ScopedWaitBudget> budget;
                if(wb_us)
                    budget.reset(new Transactional::ScopedWaitBudget(
                        (int64_t)wb_us));
                //! Counted inside the lambda, because iterate_commit re-runs
                //! it on every conflict: attempts == 1 on a slow commit means
                //! one long pass, not a retry storm.
                std::uint64_t attempts = 0;
                //! Sampled BEFORE the clock starts, and after it stops.  These
                //! are other threads' counters, written continuously from other
                //! cores, so each load is a cross-core transfer on a line that
                //! is essentially never in this core's cache — the measurement
                //! would otherwise include the instrument's own coherence
                //! traffic and scale with the number of roles.  The cost of
                //! moving them out is a few hundred nanoseconds of slack in the
                //! "system progress during a slow commit" attribution, which is
                //! read against counts in the hundreds.
                std::uint64_t sys0 = 0;
                for(int t = T_UI; t < nthreads; ++t)
                    sys0 += progress[t].load(std::memory_order_relaxed);
#if KAME_STM_NEG_DIAG
                //! Zero the thread's counters so what we read after the commit
                //! is this commit's, not the downstream half's of the previous
                //! cycle.
                (void)Transactional::neg_diag_snapshot(true);
#endif
                const std::uint64_t t_rec = now_ns();
                std::uint64_t ph_snap = 0, ph_write = 0, ph_commit = 0,
                              ph_retry = 0, ph_retry_bu = 0;
                if( !phase_mode) {
                    devA->iterate_commit([&](Tr &tr){
                        ++attempts;
                        tr[ *devA].m_x++;
                        for(auto &l : leavesA) tr[ *l].m_x++;
                    });
                }
                else {
                    //! iterate_commit, expanded.  Its body is
                    //! `for(Transaction tr(node);; ++tr) { closure(tr);
                    //!  if(tr.commit()) ... }`, and `operator++` is private, so
                    //! the expansion goes through the public commitOrNext(),
                    //! which is commit() plus that same ++ on failure.  The
                    //! split that falls out is better than the one intended:
                    //! the FINAL, successful commit is timed on its own, and
                    //! failed attempts are charged to `retry` together with the
                    //! re-snapshot they trigger.  With slow commits averaging
                    //! 2.1 attempts, that separates "the commit that worked was
                    //! slow" from "the ones that did not were".
                    const std::uint64_t t0 = now_ns();
                    Tr tr( *devA);
                    std::uint64_t t1 = now_ns();
                    ph_snap += t1 - t0;
                    for(;;) {
                        ++attempts;
                        tr[ *devA].m_x++;
                        for(auto &l : leavesA) tr[ *l].m_x++;
                        const std::uint64_t t2 = now_ns();
                        ph_write += t2 - t1;
#if KAME_STM_NEG_DIAG
                        //! Differenced across exactly the boundaries `retry`
                        //! is, so the share below is of the retry segment and
                        //! not of the commit.  \sa PhaseStat::rbu
                        const std::uint64_t bu0 = bundle_unbundle_ns();
#endif
                        const bool ok = tr.commitOrNext();
                        t1 = now_ns();
                        if(ok) { ph_commit += t1 - t2; break; }
                        ph_retry += t1 - t2;
#if KAME_STM_NEG_DIAG
                        ph_retry_bu += bundle_unbundle_ns() - bu0;
#endif
                    }
                }
                const std::uint64_t t_end = now_ns();
                const std::uint64_t dt_rec = t_end - t_rec;
                const bool warm = (t_end >= t_warm_end);
                if(warm) {
#ifndef DISABLE_POOL_ALLOCATOR
                    if( !rt_warm_sampled) {
                        rt_warm_sampled = true;
                        rt_viol_warm    = kame_pool_rt_violations();
                        rt_reclaim_warm = kame_pool_rt_deferred_reclaims();
                        rt_unmap_warm   = kame_pool_rt_deferred_unmaps();
                        rt_bytes_warm   = kame_pool_reserved_bytes();
                    }
#endif
#if defined(__linux__)
                    if(rt_warm_first) {
                        rt_warm_first = false;
                        struct rusage ru;
                        if(getrusage(RUSAGE_THREAD, &ru) == 0) {
                            ru_min_warm = ru.ru_minflt; ru_maj_warm = ru.ru_majflt;
                            ru_nvcsw_warm = ru.ru_nvcsw; ru_nivcsw_warm = ru.ru_nivcsw;
                        }
                    }
#endif
                    //! WHEN the extremes happen, not just how big they are.
                    //! Without this the histogram cannot distinguish a tail
                    //! that is spread over the run from one that is warm-up
                    //! residue the warm-up window failed to cover — and a MAX
                    //! is one sample, so "is 500 ms enough?" is otherwise
                    //! unanswerable except by re-running with a bigger window
                    //! and hoping the difference is not noise.
                    if(dt_rec > acq_max_seen) {
                        acq_max_seen = dt_rec;
                        acq_max_at   = t_end - t_warm_end;
                    }
                    if((dt_rec >= (std::uint64_t)slow_ns)
                            && (t_end - t_warm_end < 1000000000ull))
                        acq_slow_1st_sec++;
                    acq_hist.add(dt_rec);
                    std::uint64_t sys1 = 0;   // …and here, after the clock.
                    for(int t = T_UI; t < nthreads; ++t)
                        sys1 += progress[t].load(std::memory_order_relaxed);
                    acq_retries.add(dt_rec, attempts, (std::uint64_t)slow_ns,
                                    sys1 - sys0);
                    if(phase_mode && (dt_rec >= (std::uint64_t)slow_ns))
                        phase_slow.add(dt_rec, ph_snap, ph_write, ph_commit,
                                       ph_retry, ph_retry_bu);
#if KAME_STM_NEG_DIAG
                    {   const auto d = Transactional::neg_diag_snapshot(false);
                        ll_all.add(d);
                        if(dt_rec >= (std::uint64_t)slow_ns)
                            slow_diag.add(dt_rec, d);
                    }
#endif
                }
                else     cold_n.fetch_add(1, std::memory_order_relaxed);
#if defined(__linux__)
                if(warm && (tracing_on_fd >= 0) &&
                   (dt_rec >= (std::uint64_t)trace_us * 1000ull) &&
                   !trace_fired.exchange(true, std::memory_order_relaxed)) {
                    char m[96];
                    int n = std::snprintf(m, sizeof(m),
                        "KAME_MIX: record commit took %llu ns\n",
                        (unsigned long long)dt_rec);
                    ssize_t w = ::write(trace_marker_fd, m, (size_t)n);
                    w = ::write(tracing_on_fd, "0\n", 2);   // freeze the buffer
                    (void)w;
                }
#endif
                progress[T_HIGHEST].fetch_add(1, std::memory_order_relaxed);
                // the demoted downstream: entry writes + visualize snapshot.
                Transactional::ScopedDemoteRealtime _demoted;
                lA(0)->iterate_commit([&](Tr &tr){
                    tr[ *lA(0)].m_x++;
                });
                {
                    Ss shot( *devA);
                    (void)shot[ *lA(1)].m_x;
                }
                progress[T_DOWNSTREAM].fetch_add(1, std::memory_order_relaxed);
            }
#ifndef DISABLE_POOL_ALLOCATOR
            //! Precondition 5: drain from a NON-critical phase.  This is that
            //! phase — after the record commit, outside its wait budget — and
            //! it is the syscall batch the per-thread gate keeps off the
            //! deadline path.  Without it the deferred work is not eliminated,
            //! only postponed until something else pays for it.
            if((rt_pool >= 2) && (rt_drain_every > 0)
                    && (++acq_iter % (std::uint64_t)rt_drain_every == 0))
                kame_pool_rt_drain();
#endif
            pause_us(hi_duty_us);
        }
#if defined(__linux__)
        {   struct rusage ru;
            if(getrusage(RUSAGE_THREAD, &ru) == 0) {
                ru_min_end = ru.ru_minflt; ru_maj_end = ru.ru_majflt;
                ru_nvcsw_end = ru.ru_nvcsw; ru_nivcsw_end = ru.ru_nivcsw;
            }
        }
#endif
    });

    // --- UI_DEFERRABLE: the main-thread mix.
    ts.emplace_back([&]{
        if(os_starve >= 1) os_be_starved(); else os_be_ordinary();
        os_pin(cpu_house);
        Transactional::ScopedPriority pr(
            Transactional::Priority::UI_DEFERRABLE);
        uint64_t i = 0;
        while( !stop.load(std::memory_order_relaxed)) {
            ++i;
            {   // graph redraw: root Snapshot — bundles the whole tree.
                Ss shot( *root);
                (void)shot[ *devA].m_x;
            }
            // widget edit: leaf write.
            leavesP[i % leavesP.size()]->iterate_commit([&](Tr &tr){
                tr[ *leavesP[i % leavesP.size()]].m_x++;
            });
            if(i % (uint64_t)ui_wide == 0) {
                // settings apply: root-scope transaction.
                root->iterate_commit([&](Tr &tr){
                    tr[ *panel].m_x++;
                    tr[ *devB].m_x++;
                });
            }
            if(i % 16 == 0) {
                // the classic NMR trigger: a settings write into the
                // MEASURING driver's own subtree, mid-acquisition.
                lA(2)->iterate_commit([&](Tr &tr){
                    tr[ *lA(2)].m_x++;
                });
            }
            if(i % 32 == 0) {
                // tool/driver creation & removal: structural churn.
                shared_ptr<MyNode> tmp(MyNode::create<MyNode>());
                panel->insert(tmp);
                panel->release(tmp);
            }
            progress[T_UI].fetch_add(1, std::memory_order_relaxed);
            pause_us(ui_period_us);
        }
    });

    // --- SCRIPTING: the Python thread's shape — wide snapshots (reading
    // scalar entries / node tree) plus occasional writes at driver and panel
    // scope.  The second lowprio thread the field always has.
    for(long k = 0; k < n_scripting; ++k) {
        ts.emplace_back([&, k]{
            if(os_starve >= 1) os_be_starved(); else os_be_ordinary();
            os_pin(cpu_house);
            Transactional::ScopedPriority pr(
                Transactional::Priority::SCRIPTING);
            uint64_t i = 0;
            while( !stop.load(std::memory_order_relaxed)) {
                ++i;
                {
                    Ss shot( *root);           // read the tree, like a script
                    (void)shot[ *devA].m_x;
                }
                if(i % 4 == 0)
                    lA(3)->iterate_commit([&](Tr &tr){
                        tr[ *lA(3)].m_x++;    // script pokes the driver
                    });
                if(i % 16 == 0)
                    panel->iterate_commit([&](Tr &tr){
                        tr[ *leavesP[(i / 16 + (uint64_t)k) % leavesP.size()]].m_x++;
                    });
                progress[T_SCRIPT0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- NORMAL: other drivers on their own subtree.
    for(long k = 0; k < n_normals; ++k) {
        ts.emplace_back([&, k]{
            //! Only at starve level 2.  A starved NORMAL holder is the
            //! unbounded case — its privilege never expires and the wait
            //! behind it is budget-exempt — so it is kept behind its own
            //! level rather than riding along with the revocable tiers.
            if(os_starve >= 2) os_be_starved(); else os_be_ordinary();
            os_pin(cpu_house);
            Transactional::ScopedPriority pr(Transactional::Priority::NORMAL);
            uint64_t i = 0;
            while( !stop.load(std::memory_order_relaxed)) {
                ++i;
                if(normal_xsub && ((i % 4) == 0)) {
                    //! XSecondaryDriver's shape: a NORMAL transaction whose
                    //! scope SPANS the acquiring driver's subtree, because it
                    //! reads the primary's record and writes its own result.
                    //! Without this role the NORMAL peers only ever touch
                    //! devB, so they can never hold privilege on the linkage
                    //! the acquisition thread commits to — and the
                    //! never-expiring, budget-exempt case cannot arise no
                    //! matter how hard they are starved.  This is the role in
                    //! the 2026-07-30 field crash that transaction_priv_expiry
                    //! _test reproduces white-box.
                    root->iterate_commit([&](Tr &tr){
                        (void)tr[ *devA].m_x;
                        tr[ *leavesB[(i + (uint64_t)k) % leavesB.size()]].m_x++;
                    });
                }
                else {
                    devB->iterate_commit([&](Tr &tr){
                        tr[ *leavesB[(i + (uint64_t)k) % leavesB.size()]].m_x++;
                    });
                }
                progress[T_NORMAL0 + (size_t)k].fetch_add(
                    1, std::memory_order_relaxed);
            }
        });
    }

    // --- Housekeeping-core load.  SCHED_IDLE only starves a thread when
    // something SCHED_OTHER wants the same CPU; at level 1 the NORMAL peers
    // supply that, but at level 2 they are idled too and every thread on the
    // housekeeping CPU would run freely again.  These spinners are what makes
    // "the housekeeping core is saturated" true.  SCHED_OTHER, never FIFO, and
    // pinned away from the acquisition CPU.
    const long n_load = (os_starve >= 2) ? env_long("KAME_MIX_OS_LOAD", 2) : 0;
    for(long k = 0; k < n_load; ++k) {
        ts.emplace_back([&]{
            os_be_ordinary();
            os_pin(cpu_house);
            volatile uint64_t sink = 0;
            while( !stop.load(std::memory_order_relaxed))
                for(int i = 0; i < 4096; ++i) sink = sink + i;
            (void)sink;
        });
    }
    if(n_load)
        std::printf("  OS arm: +%ld SCHED_OTHER spinner(s) on cpu%ld\n",
                    n_load, cpu_house);

    // --- watchdog: stall = livelock.
    static const char *kNames[] = {"acq(record)", "  demoted downstream",
                                   "UI_DEFERRABLE", "SCRIPTING", "NORMAL"};
    auto name_of = [&](int t){
        return kNames[t < T_SCRIPT0 ? t : (t < T_NORMAL0 ? 3 : 4)]; };
    std::vector<uint64_t> last(nthreads, 0), last_change_ms(nthreads, 0);
    int failures = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto ms_now = [&]{ return (uint64_t)
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count(); };
    while((long)ms_now() < secs * 1000 && !failures) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        uint64_t now_ms = ms_now();
        for(int t = 0; t < nthreads; ++t) {
            uint64_t v = progress[t].load(std::memory_order_relaxed);
            if(v != last[t]) { last[t] = v; last_change_ms[t] = now_ms; }
            else if(now_ms - last_change_ms[t] >= (uint64_t)stall_secs * 1000) {
                std::printf("STALL: thread %d (%s) made no progress for "
                            "%llu ms at count=%llu — livelock.\n",
                            t, name_of(t),
                            (unsigned long long)(now_ms - last_change_ms[t]),
                            (unsigned long long)v);
                ++failures;
            }
        }
    }
    stop.store(true);
    for(auto &t : ts) t.join();

    double el = ms_now() / 1000.0;
    for(int t = 0; t < nthreads; ++t)
        std::printf("  %-24s %12llu commits  (%.0f /s)\n", name_of(t),
                    (unsigned long long)progress[t].load(),
                    progress[t].load() / el);
    std::printf("  priv strips (Rule 0): %llu\n",
        (unsigned long long)Transactional::detail::g_priv_strips.load());

    // The realtime question: not "did it keep up" but "did any one record take
    // too long".  Percentiles are printed only where the sample count can
    // support them.
    std::printf("  acq record-commit latency  (warm; %llu cold commit(s) in the "
                "first %ld ms dropped)\n", (unsigned long long)cold_n.load(),
                warmup_ms);
    std::printf("    n=%llu  mean=%llu ns  p50=%llu",
                (unsigned long long)acq_hist.n,
                (unsigned long long)(acq_hist.n ? acq_hist.sum / acq_hist.n : 0),
                (unsigned long long)acq_hist.pct(0.50));
    static const double kP[] = {0.99, 0.999, 0.9999, 0.99999};
    static const char *kPN[] = {"p99", "p99.9", "p99.99", "p99.999"};
    for(int i = 0; i < 4; ++i)
        if(acq_hist.supports(kP[i]))
            std::printf(" %s=%llu", kPN[i],
                        (unsigned long long)acq_hist.pct(kP[i]));
    std::printf("  MAX=%llu ns\n", (unsigned long long)acq_hist.max);
    //! Is the tail warm-up residue?  A MAX in the first second after the
    //! warm-up window says the window was too short and the headline number is
    //! an artefact; a MAX at t=97 s says the tail is a property of the steady
    //! state.  One sample cannot be read without knowing which, and raising
    //! KAME_MIX_WARMUP_MS until the number moves is guesswork against noise.
    //! Compared against the UNIFORM expectation for the warm window's own
    //! length, not a fixed fraction: over 19.5 warm seconds a flat tail puts
    //! 5 % of itself in the first second, and a fixed ">5 % means front-loaded"
    //! test therefore fires on flat data.  (It did, on the first run of this
    //! very line.)  The ratio is what carries meaning.
    if(acq_hist.n) {
        const double warm_s = (double)secs - (double)warmup_ms / 1000.0;
        const double expect = (warm_s > 1.0)
            ? (double)acq_retries.slow_n / warm_s : 0.0;
        const double ratio = (expect > 0.0)
            ? (double)acq_slow_1st_sec / expect : 0.0;
        std::printf("    MAX occurred %.1f s after warm-up ended; %llu of %llu "
                    "slow commit(s) in the first warm second = %.2fx uniform%s\n",
                    (double)acq_max_at / 1e9,
                    (unsigned long long)acq_slow_1st_sec,
                    (unsigned long long)acq_retries.slow_n, ratio,
                    (ratio >= 3.0)
                        ? "  <= FRONT-LOADED: raise KAME_MIX_WARMUP_MS" : "");
    }
    if(phase_slow.n) {
        const double N = (double)phase_slow.n;
        std::printf("    PHASE split of the slow population (n=%llu), per commit:\n"
                    "      snapshot=%.0f  payload-write=%.0f  final commit=%.0f  failed attempts+resnap=%.0f  (ns)\n"
                    "      the MAX commit (%llu ns): snapshot=%llu write=%llu "
                    "final=%llu retry=%llu  (sum %llu, unattributed %lld)\n",
                    (unsigned long long)phase_slow.n,
                    phase_slow.snap / N, phase_slow.write / N,
                    phase_slow.commit / N, phase_slow.retry / N,
                    (unsigned long long)phase_slow.max_dt,
                    (unsigned long long)phase_slow.max_snap,
                    (unsigned long long)phase_slow.max_write,
                    (unsigned long long)phase_slow.max_commit,
                    (unsigned long long)phase_slow.max_retry,
                    (unsigned long long)(phase_slow.max_snap
                        + phase_slow.max_write + phase_slow.max_commit
                        + phase_slow.max_retry),
                    (long long)phase_slow.max_dt
                        - (long long)(phase_slow.max_snap
                        + phase_slow.max_write + phase_slow.max_commit
                        + phase_slow.max_retry));
#if KAME_STM_NEG_DIAG
        //! The multiplication the retry-path attribution never did.  "A failed
        //! attempt re-bundles the subtree and discards it" is only an
        //! explanation if N passes x the cost of a pass reaches the retry
        //! phase; done by hand from the numbers already published it fell
        //! short by 3.2x, so do it here where both terms are measured in the
        //! SAME run and cannot be paired across regimes.
        {   const double retry = phase_slow.retry / N;
            const double rbu = phase_slow.rbu / N;
            std::printf("      of that retry phase, bundle+unbundle is "
                        "%.0f ns of %.0f ns = %.0f %%  (worst commit: "
                        "%llu of %llu)%s\n",
                        rbu, retry, retry > 0 ? 100.0 * rbu / retry : 0.0,
                        (unsigned long long)phase_slow.max_rbu,
                        (unsigned long long)phase_slow.max_retry,
                        (retry > 0 && rbu < 0.5 * retry)
                            ? "\n      => does NOT close: re-bundling is a "
                              "MINORITY of the retry cost" : "");
            if(slow_diag.n) {
                const double S = (double)slow_diag.n;
                std::printf("      whole-commit bundle/unbundle for scale: "
                            "bundle %.0f ns in %.2f pass(es) (%.1f levels "
                            "each), unbundle %.0f ns in %.2f (%.1f levels)\n",
                            slow_diag.bundle_ns / S, slow_diag.bundle_calls / S,
                            slow_diag.bundle_calls
                                ? (double)slow_diag.bundle_all
                                    / (double)slow_diag.bundle_calls : 0.0,
                            slow_diag.unbundle_ns / S,
                            slow_diag.unbundle_calls / S,
                            slow_diag.unbundle_calls
                                ? (double)slow_diag.unbundle_all
                                    / (double)slow_diag.unbundle_calls : 0.0);
            }
        }
#endif
    }
#if defined(__linux__)
    //! Kernel entries on the acquisition thread over the measured window.
    //! minflt is the one that matters: a minor fault is not a syscall, so no
    //! amount of reasoning about which syscalls HIGHEST can reach excludes it,
    //! and at a few microseconds each it is the right order for the tail.
    if(ru_min_end || ru_maj_end || ru_nvcsw_end || ru_nivcsw_end) {
        const long f = ru_min_end - ru_min_warm, mj = ru_maj_end - ru_maj_warm;
        //! Scope: the whole acquisition CYCLE, record commit plus the demoted
        //! downstream, because getrusage is itself a syscall and cannot be
        //! called per commit.  So voluntary switches are EXPECTED and are the
        //! downstream half at NORMAL reaching negotiate_sleep's futex — the
        //! record commit's own sleeps are known to be zero independently, from
        //! the NegDiag block.  The faults are the number with no such
        //! alternative source, and the reason this line exists.
        std::printf("  acq thread kernel entries per cycle (record + demoted "
                    "downstream):\n    minor faults=%ld major=%ld  "
                    "ctxt sw: vol=%ld (downstream futex) invol=%ld%s\n",
                    f, mj, ru_nvcsw_end - ru_nvcsw_warm,
                    ru_nivcsw_end - ru_nivcsw_warm,
                    (f || mj) ? "   <= FAULTS ON THE MEASURED PATH" : "");
    }
#endif
#ifndef DISABLE_POOL_ALLOCATOR
    //! Precondition 4, checked rather than assumed.  `rt_violations` counts
    //! the times a realtime thread actually entered the kernel for a NEW
    //! mapping — the one event the contract says must not happen inside the
    //! section, and the one prewarm alone does NOT prevent, because prewarm
    //! provisions size classes and mapping is about address space.  Sampled
    //! from the close of the warm window so start-up mapping is excluded.
    //! Process-wide, so a peer's growth counts too — which is the right
    //! scope here, since a peer's mmap stalls this core as readily as ours.
    if(rt_warm_sampled) {
        unsigned long long v = kame_pool_rt_violations() - rt_viol_warm;
        std::size_t b = kame_pool_reserved_bytes();
        std::printf("  pool during the measured window: rt_violations=%llu  "
                    "mapped %zu -> %zu MiB (%+lld)\n",
                    v, rt_bytes_warm >> 20, b >> 20,
                    (long long)((long long)(b >> 20)
                                - (long long)(rt_bytes_warm >> 20)));
        std::printf("    deferred: reclaims +%llu  unmaps +%llu  "
                    "pending %zu MiB%s\n",
                    kame_pool_rt_deferred_reclaims() - rt_reclaim_warm,
                    kame_pool_rt_deferred_unmaps() - rt_unmap_warm,
                    kame_pool_rt_pending_bytes() >> 20,
                    v ? "   <= THE SECTION MAPPED: raise KAME_MIX_RESERVE_MIB"
                      : "");
    }
#endif
    //! How many commits reached the budget.  Without this the tail is
    //! ambiguous: a MAX that sits ON the budget can be a distribution that
    //! happens to end there or a distribution CLIPPED there, and only the
    //! second means "the budget is what you are measuring, lower it and the
    //! number follows".  Bucketed, so it is a floor on the true count.
    if(wb_us && acq_hist.n) {
        std::uint64_t clipped =
            acq_hist.at_or_above((std::uint64_t)wb_us * 1000ull);
        std::printf("    reached the %ld us budget: >=%llu commit(s) "
                    "(%.4f %%)%s\n", wb_us, (unsigned long long)clipped,
                    100.0 * (double)clipped / (double)acq_hist.n,
                    clipped ? "  <= the MAX above is the BUDGET, not the STM"
                            : "");
    }
    std::printf("    attempts/commit: all=%.3f   slow(>=%ld ns): n=%llu "
                "mean=%.3f max=%llu\n",
                acq_retries.all_n
                    ? (double)acq_retries.all_attempts / (double)acq_retries.all_n : 0.0,
                slow_ns, (unsigned long long)acq_retries.slow_n,
                acq_retries.slow_n
                    ? (double)acq_retries.slow_attempts / (double)acq_retries.slow_n : 0.0,
                (unsigned long long)acq_retries.slow_max);
#if KAME_STM_NEG_DIAG
    if(slow_diag.n) {
        const double N = (double)slow_diag.n;
        std::printf("    slow-commit negotiator breakdown (n=%llu):\n"
                    "      per commit: entries=%.2f rounds=%.2f "
                    "(exempt=%.2f) sleeps=%.2f (priv=%.2f) spins=%.2f\n"
                    "      per commit: slept=%.0f ns (exempt=%.0f, %.1f %%)  "
                    "requested=%.0f ns  spin=%.0f ns\n"
                    "      worst SINGLE wait overshoot (actual-requested) "
                    "over all slow commits: %llu ns\n"
                    "      deadline-tail spin: %.2f /commit, %.0f ns/commit\n"
                    "      INNER CAS retries (invisible to attempts): "
                    "commit=%.2f bundle=%.2f snapshot=%.2f per commit\n",
                    (unsigned long long)slow_diag.n,
                    slow_diag.entries / N, slow_diag.rounds / N,
                    slow_diag.rounds_exempt / N, slow_diag.sleeps / N,
                    slow_diag.sleeps_priv / N, slow_diag.spins / N,
                    slow_diag.slept_ns / N, slow_diag.slept_exempt_ns / N,
                    slow_diag.slept_ns
                        ? 100.0 * (double)slow_diag.slept_exempt_ns
                                / (double)slow_diag.slept_ns : 0.0,
                    slow_diag.req_ns / N, slow_diag.spin_ns / N,
                    (unsigned long long)slow_diag.late_max_ns,
                    slow_diag.tail_spins / N, slow_diag.tail_spin_ns / N,
                    slow_diag.commit_cas / N, slow_diag.bundle_cas / N,
                    slow_diag.snap_cas / N);
        const auto &m = slow_diag.max_d;
        std::printf("      the MAX commit itself (%llu ns): entries=%llu "
                    "rounds=%llu (exempt=%llu) sleeps=%llu slept=%llu ns "
                    "(exempt=%llu) requested=%llu ns spin=%llu ns\n"
                    "  commit_cas=%llu bundle_cas=%llu snapshot_cas=%llu\n"
                    "        unaccounted = %lld ns\n",
                    (unsigned long long)slow_diag.max_dt,
                    (unsigned long long)m.entries,
                    (unsigned long long)m.rounds,
                    (unsigned long long)m.rounds_exempt,
                    (unsigned long long)m.sleeps,
                    (unsigned long long)m.slept_ns,
                    (unsigned long long)m.slept_exempt_ns,
                    (unsigned long long)m.req_ns,
                    (unsigned long long)m.spin_ns,
                    (unsigned long long)m.commit_cas_retries,
                    (unsigned long long)m.bundle_cas_retries,
                    (unsigned long long)m.snapshot_retries,
                    (long long)slow_diag.max_dt - (long long)m.slept_ns
                        - (long long)m.spin_ns);
        std::printf("      livelock probe during those slow commits: "
                    "ticks=%.2f (reset=%.2f no_tags=%.2f few_retries=%.2f) "
                    "VERDICTS=%.4f  priv: tries=%.4f grants=%.4f /commit\n",
                    slow_diag.ll_ticks / N, slow_diag.ll_resets / N,
                    slow_diag.ll_no_tags / N, slow_diag.ll_few_retries / N,
                    slow_diag.ll_verdicts / N,
                    slow_diag.priv_tries / N, slow_diag.priv_grants / N);
        if(slow_diag.ll_ticks)
            std::printf("      retry margin at those ticks: my_tx_retries "
                        "mean=%.2f max=%llu  vs threshold max=%llu\n",
                        (double)slow_diag.ll_retry_sum
                            / (double)slow_diag.ll_ticks,
                        (unsigned long long)slow_diag.ll_retry_max,
                        (unsigned long long)slow_diag.ll_thresh_max);
    }
    //! The privilege chain, end to end, over the whole warm run.  Every claim
    //! goes through `if(_ll_saw && !registered)` — no priority term, so
    //! HIGHEST claims on the same terms as anyone, and `priv strips = 0` means
    //! the chain broke somewhere BEFORE the gate.  Each column is one of the
    //! three AND-ed conditions inside livelock_probe_tx_tick, so the first
    //! large one is the answer:
    //!
    //!   ticks           the probe ran at all
    //!    - reset        ... and returned false because the linkage CHANGED:
    //!                   the probe state holds ONE linkage_id, and a
    //!                   multi-nodal commit negotiates on several, so each
    //!                   switch throws the accumulated window away
    //!    - no_tags      ... blocked by `tags_owned == tags_total`, which a
    //!                   thread that has been displaced necessarily fails —
    //!                   i.e. exactly the thread that needs the rescue
    //!    - few_retries  ... blocked by `my_tx_retries >= clamp(sig_C*2, 3,
    //!                   hardware_concurrency())` alone
    //!   VERDICTS        reached LIVELOCK, the only input to the claim gate
    //!   tries/grants    what the gate then did with it
    if(ll_all.n) {
        const double A = (double)ll_all.n;
        std::printf("    livelock probe over ALL %llu warm record commits:\n"
                    "      per commit: ticks=%.2f  -> reset=%.2f  "
                    "no_tags=%.2f  few_retries=%.2f  VERDICTS=%.6f\n"
                    "      privilege: tries=%.6f grants=%.6f per commit "
                    "(%llu / %llu absolute)\n",
                    (unsigned long long)ll_all.n,
                    ll_all.ticks / A, ll_all.resets / A, ll_all.no_tags / A,
                    ll_all.few_retries / A, ll_all.verdicts / A,
                    ll_all.tries / A, ll_all.grants / A,
                    (unsigned long long)ll_all.tries,
                    (unsigned long long)ll_all.grants);
        if(ll_all.ticks) {
            //! The decisive one.  `few_retries` being the largest blocker only
            //! says the threshold binds; this says whether it binds by a
            //! hair or by a mile — and whether it is reachable AT ALL.  Do
            //! not derive this from the outer attempt count: snapshot()'s
            //! retry loop bumps the same counter live and restores it on
            //! scope exit, so a tick from inside it sees more retries than
            //! any number of attempts would predict.
            std::printf("      retry margin over the run: my_tx_retries "
                        "mean=%.2f max=%llu  vs threshold "
                        "clamp(sig_C*2,3,nproc) max=%llu  =>  %s\n",
                        (double)ll_all.retry_sum / (double)ll_all.ticks,
                        (unsigned long long)ll_all.retry_max,
                        (unsigned long long)ll_all.thresh_max,
                        ll_all.retry_max >= ll_all.thresh_max
                            ? "REACHABLE — the threshold is met sometimes, so "
                              "the other two conditions decide"
                            : "UNREACHABLE — no tick in this run ever had "
                              "enough retries, at any contention level");
        }
        if( !ll_all.ticks)
            std::printf("      => the probe NEVER RAN.  Privilege cannot fire "
                        "on any path; the tail is not a privilege failure but "
                        "a privilege ABSENCE.\n");
        else if( !ll_all.verdicts)
            std::printf("      => the probe ran %llu times and never reached a "
                        "verdict.  The largest blocked column above is the "
                        "reason privilege never fires.\n",
                        (unsigned long long)ll_all.ticks);
    }
#endif
    std::printf("    other roles' commits DURING a slow one: mean=%llu max=%llu"
                "  (~0 => the holder was stuck; large => they progressed and "
                "this thread kept losing)\n",
                (unsigned long long)(acq_retries.slow_n
                    ? acq_retries.slow_sys / acq_retries.slow_n : 0),
                (unsigned long long)acq_retries.slow_sys_max);
    if(deadline_us > 0) {
        const std::uint64_t over =
            acq_hist.at_or_above((std::uint64_t)deadline_us * 1000ull);
        std::printf("  over the %ld us deadline: %llu of %llu\n",
                    deadline_us, (unsigned long long)over,
                    (unsigned long long)acq_hist.n);
        if(over) ++failures;
    }
    else {
        std::printf("  (no deadline asserted; set KAME_MIX_DEADLINE_US to make "
                    "the MAX above a pass/fail — and quote it against the "
                    "host's own floor, e.g. cyclictest max)\n");
    }
#if defined(__linux__)
    if(tracing_on_fd >= 0) {
        std::printf(trace_fired.load()
            ? "  breaktrace FIRED — tracing is off and the buffer holds the "
              "run-up; read it with `cat /sys/kernel/tracing/trace`, then "
              "`echo 1 > .../tracing_on` to re-arm.\n"
            : "  breaktrace did not fire (no commit reached %ld us).\n",
            trace_us);
        ::close(tracing_on_fd);
        ::close(trace_marker_fd);
    }
#endif
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
