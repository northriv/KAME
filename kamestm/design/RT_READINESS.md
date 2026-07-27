# kamestm — RT readiness: S0, the measured baseline

Making the STM real-time starts the way the allocator's programme should have
(`kamepoolalloc/design/RT_READINESS.md`): **measure first**. There the harness
came late, and the cost was a mis-sized problem, a mechanism that turned out
weaker than claimed, and a 4 % regression found several commits after it landed.
So nothing is changed in `kamestm/` yet — this records what the STM does today.

Two standing constraints on everything that follows:

* **Do not regress the non-realtime path.** The allocator taught this
  concretely: a single `thread_local` read added to a hot path cost 4 % of
  cross-thread free throughput, because on a macOS dylib that read is a
  `_tlv_get_addr` call. Any realtime mechanism must be a *parameterisation of
  existing structure*, not a branch on the fast path.
* **No realtime-only design.** The target is that `Priority::NORMAL` becomes
  bounded — not that a privileged class gets an escape hatch.
  `Priority::HIGHEST` already has one (`transaction_neg_impl.h:1332`,
  `if(entry_pr == Priority::HIGHEST) break;` at the head of the wait loop), and
  that is exactly what does *not* solve the problem.

## The harness

`tests/transaction_latency_bench.cpp` — per-commit latency, max and
percentiles, never a mean. Pure observation: it timestamps around
`iterate_commit` from outside and adds nothing to `kamestm`, so it cannot bias
what it measures or regress anything. Workload shapes mirror
`transaction_payload_integrity_3level_mixed_test` (leaf / grand / mixed), so
these latency figures line up with that test's throughput figures.

Not a ctest: absolute latencies are machine-specific. Run it deliberately.

## Baseline (Apple M3, Release, 3 s timed after 0.5 s warmup)

    threads=4, grand%=10                p50      p99.9    p99.999      MAX
      leaf   (own node)                 128 ns   1536 ns    24.6 us   677 us
      grand  (3-level bundle)           256 ns    768 ns    25.2 ms   147 ms
      mixed  (10 % grand)               128 ns   1280 ns     6.3 ms    17.9 ms

**The ordinary path is already fast**: p50 128–256 ns, p99.9 under 1.6 µs, at
tens of millions of commits/s. The problem is entirely in the deep tail.

## What the tail is NOT

**Not the sleep chunk size — and the reason matters.** A contender that loses
the spin band waits on a condition variable for `KAME_NEG_SLEEP_US_PER_MS` µs
(compile-time, default 1000). Rebuilt at 250 µs — a 4× cut — and the tail did
not move at all: grand p99.999 stayed at `25,165,824 ns` and mixed
p99.99/p99.999 at `2,621,440` / `6,291,456`, *bit-identical*.

The first reading of that was "so the sleep is not involved", and it was
**wrong**. The wait does not end when a chunk expires — it ends when the
contended node becomes winnable. Shrinking the chunk therefore changes the
polling granularity (4× as many, 4× shorter) and not the wall time. The sleep
is still the mechanism; what sets the duration is **how long the transaction is
denied its turn**. Recorded because the wrong inference points somewhere very
different (tune the chunk / replace it with a spin) from the right one (fix the
arbitration).

**Not a retry storm.** `iterate_commit` invokes its lambda once per attempt, so
the retry count is observable with no library change. Counted per commit and
correlated with its latency: **attempts/commit = 1.000** overall, and for
commits ≥ 100 µs the mean is 1.000 (leaf) / 1.264 (grand) / 1.445 (mixed),
max 1–10. So even the slowest commits mostly finish in ONE attempt — the time
goes *inside* a single attempt, which is what localises it to negotiation.

**Not the bundle work.** The grand arm is the 3-level bundle,
`1 + 2(N+1)` CAS per commit — the expensive shape. Its p50/p99.9 (256/768 ns)
are tight and scale gently with threads. The work is not where the time goes.

## What the tail IS — arbitration starvation

The grand arm against thread count (2 s each, contention is the only variable):

    threads    p50      p99.9    p99.99     p99.999      MAX
      1        192 ns    448 ns    6.1 us    28.7 us    15.8 ms
      2        192 ns    512 ns    7.2 us     7.3 ms    69.9 ms
      4        256 ns    640 ns   16.4 us    21.0 ms   159.9 ms
      8        384 ns   1024 ns    3.7 ms    67.1 ms   232.9 ms

p50 and p99.9 barely move (2× across 8× the threads). **p99.999 moves 2300×**,
29 µs → 67 ms, monotonically with contention. That is a losing transaction
waiting its turn — the fairness tail — not work and not the sleep primitive.

Consistent with the earlier finding that the mixed-scope penalty is bundle
churn plus *negotiation waiting* rather than a retry storm: the waiting is now
quantified, and it is the whole story at the tail.

### S1' — diagnosed: the escape hatch never arms

Instrumented with `-DKAME_STM_NEG_DIAG=1` (compile-time, default off, purely a
diagnostic — nothing in the library behaves differently when it is on, and it
is kept separate from the non-throughput-neutral `KAME_ADAPT_INSTRUMENT`).
Counters are snapshot per commit and accumulated for slow ones only.

Slow commits (≥ 100 µs), 8 threads:

    arm      n      rounds/commit   sleeps/commit   slept/commit   priv tries/grants
    leaf    2436     0.00 (max 0)        0.00            0 ns         0.000 / 0.000
    grand    808     3.91 (max 17)      15.41        25.9 ms          0.000 / 0.000
    mixed   5676     3.26 (max 10)       6.27         3.7 ms          0.004 / 0.004

Three things fall out at once:

* **The leaf tail is entirely the OS.** Zero rounds, zero sleeps, zero
  negotiation — those 2436 slow commits never entered the negotiator. The
  control was right, and `MAX` on this host is not an STM number.
* **The grand tail is sleeping**, and it is essentially all of it: 25.9 ms slept
  out of a ~26 ms mean slow commit. So the earlier correction holds — the sleep
  is the mechanism, the duration is set by not being able to win.
* **Privilege is never even attempted.** `priv tries = 0.000` while waiting
  25.9 ms, on the very path whose comment calls it the "fair-mode escape".

The cause, and it is sharper than any of the three candidates guessed above.
The claim is gated on a livelock verdict:

    verdict = LIVELOCK  iff  tags_total > 0 && tags_owned == tags_total
                             && my_tx_retries >= clamp(sig_C*2, 3, hw_procs)

and the age floor (`min_privilege_age_us`, 300 µs) is checked *inside* that
branch. But the starved transaction **does not retry** — measured
attempts/commit 1.000, and 1.233 even for the slow ones. It waits inside a
single attempt. So `my_tx_retries` stays below the threshold, the verdict is
never `LIVELOCK`, the branch is never entered, and the age floor is never
evaluated at all.

> **A transaction that waits without retrying is invisible to a
> retry-counting livelock detector — so the escape hatch built for exactly
> this situation never arms.**

Every measurement follows from that: attempts 1.000, priv tries 0.000, tens of
ms slept, tail scaling with contention. Note the failure mode is specific:
under a workload with genuine CAS conflicts the detector *does* arm; it is the
waiting-dominated mode — the realtime-relevant one — that it misses.

This is also a fidelity item for the paper. The TLA+ liveness argument ranks by
an ageing tag and concludes the oldest eventually wins; the C++ only *claims*
that rank behind a retry-count detector. Whether the model's `ClearMyTags`-era
argument still covers the implementation when the promotion is unreachable is a
question the dossier should carry, separately from what the ranking proves.

### Measurement caveat, so nobody chases it

At **1 thread — no contention at all — MAX is still 15.8 ms.** Nothing in the
STM can starve a sole committer, so that is external (scheduler preemption, or
a one-off allocator region growth). **MAX is contaminated on this host; read
p99.99 / p99.999.**

The same floor shows in the *leaf* arm, and it is the control that keeps the
grand result honest: at 8 threads leaf reaches p99.999 = 786 µs with
**1.000 attempts** on every slow commit, on a node nobody else touches. That is
the OS descheduling the measuring thread, not the STM. Grand at the same thread
count is p99.999 = 84 ms — **two orders of magnitude above that floor**, which
is what makes it a real STM effect rather than noise.

## Where this leaves the plan

The step that looked obvious before measuring — replace the CV sleep with a
bounded spin — **would not have fixed the tail**, and would have cost CPU on a
machine where KAME already runs more threads than cores. It is dropped as the
first move.

    S1'  DONE — the promotion is gated on a retry-count livelock verdict that
         a purely-waiting transaction never trips.
    S2   TRIED AND REJECTED — see below.  Age-reachable promotion works
         mechanically but trades throughput for tail at every setting.
    S3   Bound the (contenders x per-commit work) product, which S2 showed is
         what actually sets the tail.
    S4   Verify the commit path is syscall-free under the allocator's
         KAME_RT_DEFER (assert kame_pool_rt_violations() == 0 in this harness).
    S5   The bound statement and contract, once there are numbers to state.

Reproduce:

```bash
cd kamestm/tests && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build -j --target transaction_latency_bench
build/transaction_latency_bench -t 4 -s 3 -x 10        # the baseline table
for t in 1 2 4 8; do build/transaction_latency_bench -t $t -s 2 -m grand; done
# chunk sweep (compile-time, so it costs a build and no runtime branch):
cmake -S . -B b250 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DKAME_NEG_SLEEP_US_PER_MS=250"
```

## S2 — tried, measured, rejected (and what it proved)

Implemented exactly as designed: reach the *existing* privilege claim by AGE as
well as by the livelock verdict (`(_ll_saw || aged)`), plus leave the chunked
sleep early once aged so the outer round — where the claim lives — is re-entered
promptly rather than every ~6.6 ms. The verdict's own condition was untouched,
so retry-driven promotion behaved exactly as before, and nothing was added to
any fast path. Deliberately no expiry: the model moves `priorityTag` only
towards an older thread and KEEPS it on success, so a rank that could lapse is
what its ranking argument cannot have.

**The mechanism fired, verifiably.** With `KAME_STM_NEG_DIAG=1`, grand slow
commits went from `priv tries 0.000 / grants 0.000` to `0.118 / 0.116`, sleeps
per commit 15.41 → 2.93, and time slept per commit 25.9 ms → 4.96 ms. That is
direct evidence the intended path was taken, not an inference from the outcome.

**And it still fails the acceptance gate.** Interleaved A/B on the established
throughput test (`transaction_payload_integrity_3level_mixed_test 3 128 10 10`),
against a worktree built at the pre-change commit:

| age floor | throughput @128t | grand p99.9 | p99.99 | p99.999 | MAX |
|---|---:|---:|---:|---:|---:|
| *pre-S2* | *9.6 M/s* | *1,024 ns* | *2.6 ms* | *83.9 ms* | *272 ms* |
| 300 µs | 4.4 M/s (**−54 %**) | 4,096 ns | 8.4 ms | 21.0 ms | 43.9 ms |
| 3 ms | 6.4 M/s (−33 %) | 2,560 ns | 10.5 ms | 21.0 ms | 40.3 ms |
| 30 ms | 8.7 M/s (−9.5 %) | 1,024 ns | 12.6 ms | 41.9 ms | 67.3 ms |

Every setting costs throughput, and p99.99 is *worse* at all of them. The
mechanism does not remove waiting — it redistributes it: mean latency (3,294 →
3,313 ns) and total work are unchanged, so cutting rare 272 ms starvation
simply makes more commits wait moderately (slow-commit count 808 → 4,232).
300 µs is strictly dominated by 3 ms (same tail, half the cost). Reverted; the
committed instrumentation stays.

### What the failure proves, which is the useful part

Perfect arbitration ordering cannot beat the product

    wait  ≈  (contenders on the node)  ×  (per-commit work)

because a promoted transaction still has to wait for the current holder, and
the holders form a chain. Fairness decides *who* waits and in what order; it
cannot reduce *how much* waiting exists. At 8–128 threads committing at
grand scope, that product is inherently tens of milliseconds, which is exactly
the tail measured — and no amount of promotion tuning moved it below 21 ms.

So the bound stated earlier is not merely a way to *describe* the wait, it is
the thing that has to be attacked:

* reduce **contenders per node** — scope, so that unrelated commits do not
  serialise on one linkage; or
* reduce **per-commit work** — bundle churn is O(subtree), so a grand-scope
  commit pays for every child.

That is S3, and S2's rejection is what establishes it is not optional. It also
sharpens the realtime contract: a bounded commit will require a bounded
*scope*, in the same way the allocator's contract required a stable working set
— a precondition on the caller, not something the arbitration can supply.

## S2 post-mortem — the waiter is invisible, and that is the real constraint

Prompted by the question "is the privilege/tag holder itself sleeping?" —
hold-and-wait would mean peers queue behind a *sleeping* holder, and the wait
would be set by the sleep rather than by anyone's work. Measured it instead of
reasoning about it, by counting, at every sleep, how many of this Tx's tagged
linkages still carry its stamp.

First answer looked like a clean refutation: **grand — 0.0 % of sleeps hold any
tag.** It was vacuous. Adding the tagged-list size to the same probe:

    arm     sleeps holding >=1 tag   tags/sleep   tagged-list size at sleep
    grand         0.0 %                 0.00              0.00
    mixed        38.2 %                 1.80              1.14

**The grand sleeper has tagged nothing at all.** So "owns none" said nothing
about hold-and-wait — there was nothing to hold. (In mixed, where leaf commits
do tag, hold-and-wait is real at 38 % of sleeps and worth revisiting.)

Why nothing is tagged closes the chain. `tag_as_contender` has exactly **one**
call site: the *pre-commit retry tag* in `operator++`, and only
`if(isMultiNodal())`. A transaction tags when it **retries**. Measured,
attempts/commit is 1.000 — the starving Tx does not retry. So:

* it never tags → `tags_total == 0`;
* the livelock verdict requires `tags_total > 0 && tags_owned == tags_total`,
  so it **can never fire, whatever the retry count** — the retry threshold I
  identified in S1' is real but is not even the first blocker;
* privilege is therefore unclaimable — and in per-linkage mode the claim
  *walks `m_tagged_linkages`*, so with an empty list there is nothing to
  upgrade even once the branch is entered (which is why S2 needed such an
  aggressive floor to move anything, and why it mostly perturbed);
* and, most important, **its age is never registered anywhere**, so the
  oldest-wins comparison that the whole design rests on never has this
  transaction as an operand.

> A transaction that waits without retrying is invisible to the fairness
> machinery. It sleeps, wakes, finds the node still taken, sleeps again —
> with nothing recording that it is waiting, or for how long.

### This corrects the S2 lesson recorded above

That section concluded the tail is set by `(contenders × per-commit work)` and
that ordering cannot beat it. That inference was drawn from S2 failing — but S2
was, for the grand arm, largely a **no-op on an empty tag list**, so its failure
is weak evidence about ordering in general. The product bound is still the right
*form* of a bound; it is not established as the binding constraint here.

### The candidate that follows, and it is a fidelity fix

Tag when a transaction **yields**, not only when it retries. The model places
its tag on failure-to-proceed and orders by age from that moment
(`TagAfterFail`: an older tag displaces a younger one); the C++ places it only
on a transaction-level retry, which is a far coarser trigger and misses the
population that blocks without ever attempting. Making the waiter visible would
give the existing machinery — oldest-wins displacement, the verdict, the age
floor — something to work with, at no fast-path cost (the yield path is already
the slow path).

Stated as a candidate, not a fix: it must be measured against the same gate that
rejected S2 (throughput at 128 threads, interleaved A/B), and hold-and-wait in
the mixed arm (38 %) says tagging more will interact with holders that sleep.
