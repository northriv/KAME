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

## S3 candidate "tag on yield" — implemented, and it is never reached

The candidate above rested on a premise I had not checked: that the starving
transaction sleeps *because a peer's tag blocks it*, so registering its own tag
at that point would make it visible to oldest-wins. Implemented it at the
obvious place — right after `fair_mode_blocks_me` in `_negotiate_internal`,
tagging only when the gate says yield.

Throughput was neutral (9.67 vs 9.73 M/s at 128 threads, interleaved — nothing
like S2's −54 %), and the tail moved from p99.999 84 → 67 ms and MAX 272 →
209 ms. Both are inside this metric's run-to-run spread, so on their own they
prove nothing — which is why the mechanism check matters:

    tagged-list size at sleep : 0.00   (unchanged)
    fair_blocks               : 0.00 per commit
    yield-site tag calls      : 0.00 per commit

**`fair_mode_blocks_me` is never true, so the site is never reached.** The
change is inert; the small tail movement was noise. Reverted.

### What that establishes

In the grand arm **no tag exists at all**, from either source: the retry tag
never fires because the Tx never retries, and the yield tag never fires because
the gate never blocks. The entire fairness machinery — tags, oldest-wins
displacement, the livelock verdict, the age floor, privilege — is **dormant**
in the workload that produces the 84 ms tail.

So the tail is not starvation in the "someone holds the node and I queue behind
them" sense. Nobody holds anything. It is produced by the **adaptive
negotiation backoff itself** deciding to sleep, ~14 times, ~25 ms in total, per
slow commit, with no arbitration involved. Three successive hypotheses —
sleep-chunk size, retry-gated promotion, tag invisibility — were each refuted
the same way: by checking whether the mechanism fired rather than whether the
number moved.

The next question is therefore about the backoff, not about fairness: **why
does the adaptive path choose to sleep for tens of milliseconds when nothing is
blocking?** That is `_neg_spin_block`'s band gate, `ms_actual`, the runner
lottery and `effective_min_runners` — and it should be attacked the same way,
by instrumenting which branch selects the sleep and with what computed
duration, before changing anything.

## Is the sleep the OS preempting us? — No. The STM asks for it.

The obvious competing explanation for a millisecond tail: the thread is not
sleeping, it is *descheduled*, and the time lands inside `cell.wait()` because
that is where it happens to be. If so there would be nothing in the STM to fix.

Settled by accumulating, alongside the measured wait, the duration actually
**requested** of `cell.wait()`:

    arm     asked / sleep   got / sleep   ratio
    grand    1,498,458 ns   1,683,828 ns  1.12x
    mixed    1,504,616 ns     284,489 ns  0.19x

**grand sleeps because it asked to.** 1.12× is ordinary wake-up latency on top
of a fully-served request, not scheduling delay. (1.5 ms per sleep, not 1 ms,
because the call site randomises `1 + (seed>>31)` ms.)

An unplanned contrast fell out of the same measurement, and it is the sharper
result:

* **mixed 0.19× — woken early four times out of five.** The notify path works
  there.
* **grand 1.12× — never woken early at all.** Every one of the ~14.5 sleeps in
  a slow commit runs to its full timeout. The `wake_one` targeting that the
  code documents as best-effort ("mis-target → natural timeout") is missing
  this sleeper every single time.

### And this finally explains the flat chunk sweep, correctly

`t_end = now_us() + ms_actual * 1000` is a **time budget**, not a chunk count.
Halving `KAME_NEG_SLEEP_US_PER_MS` therefore doubles the number of chunks
inside the same budget and leaves the total untouched — which is exactly the
bit-identical tail observed, and neither of the two explanations offered
earlier in this document (first "the sleep is not involved", then "the wait
ends when the node becomes winnable"). The second was closer but still wrong:
the wait ends when **`ms_actual` elapses**, because nothing wakes it.

So the quantity that sets the tail is `ms_actual` — how much sleep the adaptive
backoff budgets per round — together with the number of rounds, and with wakes
that never arrive to cut either short. That is what to instrument next:
which branch computes `ms_actual`, what value it produces, and why
`notify_n_contenders` never reaches a grand-arm sleeper.

## The mechanism, complete: an escalating budget that nothing cuts short

Instrumented `ms_actual` — the per-round sleep budget — alongside everything
else. Slow commits, 8 threads:

    arm     ms_actual summed/commit   max single round   actual/requested
    grand          26.51 ms                153 ms              1.12x
    mixed           3.27 ms                 52 ms              0.09x

grand's 26.51 ms of budget against ~25 ms measured sleeping: **the tail is the
backoff budget, spent in full.** And a single round can budget **153 ms**,
which is where MAX 200–380 ms comes from.

The escalation is by construction:

    ms = std::max((int)(dt2 * mult_wait / 10000), ms + 1);   // capped at 5000

`ms` grows every round — at least +1, far more via the `dt2` term — and the
round's sleep is served as ~1.5 ms `cell.wait()` chunks until `t_end = now +
ms_actual * 1000`.

### The difference between the two arms is not the backoff — it is the wake

mixed budgets the same *kind* of escalation (3.27 ms/commit, single rounds up
to 52 ms) and **serves 9 % of it** (0.09×), because wakes arrive and cut each
chunk short. grand serves **112 %** — every chunk to full timeout, never woken
once.

So the escalating budget is not by itself the defect: it is a backoff that
assumes a wake will cut it short, and that assumption holds in mixed and fails
completely in grand. Combined with the earlier finding that the grand arm has
no tags at all, the picture is consistent — a transaction that never tags is in
no target set, so `notify_n_contenders` and the tenant-verified `wake_one` have
nothing to aim at, and the budget becomes the latency.

### Two candidate directions, in preference order

1. **Make the wake reach the sleeper.** mixed is the existence proof that the
   mechanism works when it lands; the grand arm's sleeper is simply
   untargetable. This is the fix that removes the tail without touching the
   backoff's throughput role.
2. **Cap the budget when no wake can be expected.** Weaker — it trades tail for
   the spin/CPU the backoff exists to avoid, and 153 ms says the cap would have
   to be aggressive.

Both must pass the gate that rejected S2: throughput at 128 threads,
interleaved, plus a mechanism check that the wake actually lands rather than
the number merely moving.

## Candidate 1 "wake by linkage" — implemented, and it costs 75 %

The design was futex-shaped and, I still think, diagnostically right: the sleep
slot is keyed by `ProcessCounter::id()`, i.e. by *who* is waiting, so a wake
requires knowing who — and the only registries of "who is contending" (tags,
`tid_bitset`) are populated by CAS failure, which the starving transaction never
incurs. Re-keying the slot to the contended **linkage address** removes the
information asymmetry: the waker knows what it contended for, always.

Implemented exactly that — `negotiate_sleep(..., key)` selecting
`slot_of_key(linkage)`, a `wake_on_linkage()`, and a call to it beside each of
the negotiator's three existing `notify_n_contenders` sites (the tid wakes kept,
so the path that demonstrably works in the mixed arm was untouched). No new
state, ctest 9/9.

**Throughput at 128 threads: 9.5 M/s → 2.4 M/s, −75 %, 4/4 reps.** Reverted.

### Why, and what it teaches

Keying by address **concentrates** the waiters: every thread contending for the
same linkage now sleeps on the *same* cell, where before `ProcessCounter::id() %
NEGOTIATE_SLEEP_SLOTS` spread them across the slot array. At 128 threads on a
few hot linkages that turns a de-phased set of sleepers into a single hammered
userspace cell.

That is the difference between this and a real futex: a futex also keys by
address, but the queue lives in the kernel, where concentration is the design.
Here the cell is a shared userspace atomic, so **the slot spreading is
load-bearing for throughput** — it is not incidental, and any address-keyed
scheme has to preserve it.

So the tension is structural, and worth stating before the next attempt:

* to wake by *what is contended*, waiters must be findable **by** what is
  contended;
* grouping them that way concentrates them, which is what costs the throughput;
* keeping them spread (by tid) is exactly what makes them unfindable.

Resolving it needs a per-linkage **list** of waiter slots — waiters stay spread
across cells, and the linkage carries the small amount of state needed to find
them — which means adding state to `Linkage`, itself a hot structure. That is a
real design change, not a re-keying, and it should be costed before it is
attempted.

## Candidate "just place the tag" — −97 %, and it completes the picture

Fair question after the previous attempt: if the problem is that no tag exists,
place one. And it had genuinely not been tried — the earlier yield-site tag was
never reached (`fair_blocks` 0.00), so no tag was ever placed and that
experiment never actually ran.

Placed it at the site that *is* reached: unconditionally, immediately before
entering the sleep path. The expectation was specific and reasonable — the
existing tenant-verified wake matches the **linkage's stamp** against a
sleeper's published stamp, so a tag would make the sleeper findable *without*
re-keying the slots, avoiding the concentration that cost 75 %.

**Throughput at 128 threads: 9.6 M/s → 0.32 M/s. −97 %.** Reverted (9.60 M/s).

`tag_as_contender` CAS-loops on the linkage's `m_transaction_started_time`.
Turning that from a word written only on retry into one written by every
sleeper puts 128 threads on a single cache line — and once tagged, peers'
`fair_mode_blocks_me` starts returning true, so more of them yield, sleep, and
tag. The tag is rare **by design**.

### The synthesis — three failures, one cause

| attempt | what it made visible | cost |
|---|---|---|
| age-reachable promotion | the age of a waiter | −54 % |
| linkage-keyed sleep slots | which linkage a waiter waits on | −75 % |
| tag before sleeping | the waiter itself, on the linkage | −97 % |

Every one of them fails for the same reason, and it is not a tuning problem:

> **Making a waiter visible means writing shared state, and at 128 threads that
> write is the bottleneck. This STM buys its throughput by keeping waiters
> invisible — and the latency tail is exactly what that costs.**

Which is why the mechanism found earlier is self-consistent rather than a bug:
no tag → nothing for the wake to target → the escalating backoff runs to its
full budget. Each piece is doing what it was designed to do.

### What that means for the realtime goal

"`Priority::NORMAL` becomes bounded, with no throughput regression" is not
reachable by making the existing machinery fairer — the three cheapest ways to
do that cost 54–97 %. The remaining directions all change the *shape* rather
than the tuning:

1. **Waiter state that is not shared.** A per-linkage list of waiter slots
   keeps waiters spread across cells while making them findable, but it adds
   state to `Linkage` (hot) and still needs one shared write per waiter — cost
   to be measured before it is built, not after.
2. **Bound the budget instead of fixing the wake.** Cap `ms_actual`; pay in
   spin/CPU exactly what the backoff exists to save. Cheap to try, and the
   honest fallback if (1) prices out.
3. **Bound the contention instead.** Fewer contenders per linkage, or less
   per-commit work, so the wait never grows large enough to need rescuing —
   the (contenders × work) product, which is a property of how the tree is
   *used* and therefore a contract precondition rather than an STM change.

(3) is the direction the allocator's contract took, and on this evidence it is
the one most likely to survive a throughput gate.

## Candidate "mark a bit on yield" — the wake gets fixed, and the tail does not move

The remaining cheap way to become findable: not a tag (semantic, and it made
peers yield — the feedback that cost 97 %), but a **bit in a shared sleeper
registry**. `m_tid_bitset` is per-Snapshot — it records peers a thread has
*observed* — so a waiter that writes nothing observable is in nobody's set.
A shared `std::atomic<uint64_t>[8]`, one `fetch_or` before the wait and one
`fetch_and` after, carries no semantics at all: it does not make anyone yield.

Waking every recorded sleeper cost **−38 %** — the wake was indiscriminate,
since the registry knows *that* a thread sleeps but not *what for*. Waking at
most one recovered nearly all of it: **−8 %** (9.85 → 8.94 M/s). And the
mechanism check was emphatic:

    actual/requested sleep, grand:  1.12x  →  0.01x

The wake now lands, cutting 99 % of every sleep budget. **And the tail did not
move**: grand p99.999 84 → 59 ms, MAX 272 → 269 ms, p99.99 *worse*.

The breakdown says exactly why:

    grand, slow commits        before      after
      sleeps / commit           14.53    2,268.53      (156x more)
      slept / commit           24.9 ms     20.0 ms     (unchanged)
      rounds / commit            3.89        3.81      (unchanged)
      ms_actual budget         26.51 ms    21.75 ms    (unchanged)

**The thread is woken 156× more often and sleeps for the same total time.** It
wakes after 13 µs, re-checks, cannot proceed, and sleeps again.

### This settles the question the whole sequence was asking

Fixing the wake does not reduce the wait. The transaction genuinely cannot make
progress for ~20 ms; the sleep was only *how that time was spent*. Which
restores — now on much stronger evidence than the S2 rejection could give,
because here the mechanism demonstrably works and the outcome still does not
change — the conclusion:

> The tail is `(contenders on the node) × (per-commit work)`. Arbitration
> decides who waits and in what order; wake-up decides how the waiting is
> spent; **neither reduces how much waiting exists.**

Reverted: −8 % throughput and 156× the wakeups, for no latency benefit, is not
a trade worth making.

### Where this leaves the realtime goal, concretely

Five mechanisms were implemented and measured — age promotion, linkage-keyed
slots, tag-before-sleep, sleeper-registry broadcast, sleeper-registry wake-one.
Every one either cost throughput (54 / 75 / 97 / 38 / 8 %) or left the tail
where it was, and the last one did both at once while proving the wake works.

So a bounded `Priority::NORMAL` is not reachable inside the negotiator. What
remains is to attack the product itself:

* **fewer contenders per linkage** — commit at a narrower scope, so unrelated
  work does not serialise on one node;
* **less per-commit work** — bundle churn is O(subtree), so a grand-scope
  commit pays for every child.

Both are properties of *how the tree is used*, which makes a bounded commit a
**contract precondition on the caller** — a bounded scope — exactly as the
allocator's contract required a stable working set rather than a cleverer
allocator.

## "Is the holder just OS-preempted?" — no. And the answer overturns my conclusion too.

The hypothesis: the winner is descheduled while holding the node, everyone
queues behind it, and the wait is the OS rather than the STM. It fits the
evidence up to this point — no arbitration change helped, and fixing the wake
did not help, both of which are explained if the node is simply not available.

Answered without touching the library: a system-wide commit counter, sampled
before and after each measured commit, gives the one number that separates the
two worlds — **while I was waiting, did anyone else make progress?**

    arm     system commits completed DURING one slow commit
    grand         55,846  (max 616,133)
    mixed         21,017
    leaf          11,245

**The holder is not stuck.** During a ~20 ms wait the system completes ~56,000
commits: the node turns over constantly and exactly one transaction keeps
losing. (leaf is the control and confirms itself: its slow commits do zero
negotiation, so their 11,245 is what elapses while the *measuring* thread is
descheduled.)

### Which also refutes what I concluded two sections ago

I wrote that the tail is `(contenders × per-commit work)` and that neither
arbitration nor wake-up can reduce it. That is wrong, and these numbers are
what shows it: with 8 threads and a 384 ns p50, a fair turn would arrive after
~8 commits, about 3 µs. This transaction waited through **56,000**. The waiting
is not work that has to happen — it is one transaction being passed over, tens
of thousands of times.

So arbitration *is* the right lever in principle. The corrected statement is
sharper and less comfortable:

> A **wide** transaction (grand scope: bundle Parent + every child) is starved
> by **narrow** ones. It needs a quiet window across many nodes; the leaf
> commits need one node for 384 ns and never stop arriving. Nothing ever holds
> them back, so the wide one loses indefinitely.

And that is why all five mechanisms failed the throughput gate while behaving
exactly as designed: the only way to give the wide transaction its window is to
**hold the narrow ones back** — privilege, yielding, tagging all do precisely
that — and the narrow ones are where the 9.7 M commits/s comes from. The 54 /
75 / 97 / 38 / 8 % were not implementation clumsiness; they are the price of
the window, showing up in proportion to how effectively each mechanism actually
imposed it.

This is the classic wide-transaction starvation of optimistic concurrency, and
it means the realtime question is not "make the negotiator fairer" but **"how
much throughput is a bounded wide commit worth, and can the workload avoid
needing one?"** — which puts scope back at the centre, not as a fallback but as
the only lever that does not pay the tax.

## The missing piece: the timed-out sleeper never returns to try

Question that closed the loop: is the problem that a sleeper which times out
without a tag *does not return* — i.e. it re-evaluates inside the negotiator
rather than going back to attempt its CAS?

Measured, slow commits at 8 threads:

    arm      negotiator ENTRIES / commit    internal rounds / commit
    grand              1.24                          3.86
    mixed              2.81                          3.26

**Yes.** `_negotiate_internal` is entered ~1.24 times and loops ~3.9 times
inside, sleeping with an escalating budget between iterations. The timed-out
sleeper does not hand control back; it asks the gate again, later.

### Everything else follows, and two of my own conclusions were wrong

Never returning means never attempting, which means **never failing**:

* no CAS failure → `tag_as_contender` is never called → `tags_total == 0`,
  exactly as measured;
* `attempts/commit` stays 1.000, exactly as measured;
* the wake fix could not help — it wakes, the gate still refuses, it sleeps;
* arbitration could not help — **the transaction is not participating in
  arbitration at all**;
* and my reading of the 56,000 system commits was wrong too. It did not lose
  56,000 races. It was held at the **admission gate** while 56,000 commits
  that were admitted went through. There is no contention loss to be fairer
  about.

And the escalation makes it self-reinforcing:

> **Denied → back off longer → ask less often → stay denied.** The backoff
> escalates the *waiting* rather than the *applicant's priority*, which is the
> opposite of what progress requires.

mixed, with entries 2.81 against rounds 3.26, returns far more often — and has
the smaller tail. The two arms differ in exactly the predicted direction.

### The candidate this implies is different in kind from the five that failed

**After N denials (or T elapsed), return to the caller and simply attempt the
CAS.** Then either it succeeds and the wait is over, or it fails — and failing
is what tags the transaction, registers it, and engages the whole existing
fairness machinery that has been dormant for want of a single failure.

Why this does not pay the tax the other five paid: it does not hold the narrow
commits back, add shared state, or write anything on the fast path. It only
stops *suppressing* the wide transaction's attempt. The five previous
mechanisms all tried to make the wide transaction win; this one lets it play.

Risk to measure, not assume: attempting under contention is what the admission
gate exists to prevent (CAS storms). So the parameter is how long to wait
before forcing an attempt, and the gate is the same one — throughput at 128
threads, interleaved — plus the mechanism check that `attempts` and
`tags_total` actually become non-zero, which is what tells us the attempt
happened rather than the number merely moving.

## "Return after one sleep if untagged" — the first candidate that works as designed

The rule, and it is better than the "N denials or T elapsed" I had proposed:
**if this transaction holds no tag, break out of the negotiator after a single
sleep.** No new parameter — `m_tagged_linkages.empty()` is already there — and
self-limiting: no tag means it has never failed, so it goes and tries; if the
attempt fails it tags, and from that point the unchanged escalation applies.

It also cannot weaken the protection the gate exists for. A CAS storm *is* a
run of failures, and failures tag, so a storming transaction is tagged and
keeps the old path. Only the never-attempted population changes, and for it the
gate is not preventing a storm — it is preventing the one attempt that would
end the wait.

**Every predicted consequence happened.** grand, slow commits:

    metric                    before        after
    negotiator entries/rounds  1.24 / 3.86   2.54 / 2.75   (now ~1:1: it returns)
    attempts / commit          1.233         2.452 (max 8) (it attempts, and fails)
    tagged-list at sleep       0.00          0.66          (failing tags it)

Returns → attempts → fails → tags → the dormant machinery engages. This is the
first mechanism in the sequence that demonstrably does what it was designed to.

**And the latency is a redistribution, not an improvement:**

    grand          before        after
      p99.9        1,024 ns     12,288 ns   (12x WORSE)
      p99.99        2.62 ms      8.39 ms    (3.2x worse)
      p99.999      83.9 ms      41.9 ms     (2x better)
      throughput        —        −2.9 %

The cost is finally in the acceptable range — −2.9 % against 54 / 75 / 97 / 38 /
8 % for everything before it — and the deep tail halves. But p99.9 degrades 12×,
because commits that used to wait quietly now attempt, fail, and pay for the
failed attempt; and 42 ms is still nowhere near a deadline.

So it is not committed. The judgement it needs is not technical:

* **for realtime**, the worst case is what counts, and 84 → 42 ms with −2.9 %
  is the best trade found — but it is still two orders of magnitude from
  usable, so it buys direction rather than arrival;
* **for KAME as it runs today**, p99.9 is what users feel, and 1 → 12 µs is a
  real regression for a benefit nobody currently needs.

Worth noting for whoever picks this up: the rule as written fires for *every*
untagged sleeper on its first sleep. Softening it — return only after k sleeps,
or only once older than the age floor — should recover most of the p99.9 cost
while keeping the deep-tail win, and is a tuning question now that the
mechanism is known to work. That is the obvious next experiment, and the first
one in this sequence that starts from something that functions.

## Two knobs, measured separately and together — and B4 is the trade

Both shipped compile-time, **default OFF**, so the shipping build is
byte-identical to before. `KAME_STM_CLEAR_TAGS_BEFORE_SLEEP` (A) releases our
linkage tags before the chunked sleep — aimed at the measured 38 % of mixed-arm
sleeps that held tags while blocking peers. `KAME_STM_UNTAGGED_RETURN_MS` (B)
is the softened return: an untagged transaction hands control back once `ms`
reaches the knob, instead of after the first round (B=1, the aggressive form
measured earlier) or never (base).

Throughput at 128 threads (median of 3, interleaved):

    base    9.16 M/s
    A      10.12 M/s   (+10 %)
    B=4     8.87 M/s   (−3 %)
    A+B     6.59 M/s   (−28 %)  — the knobs are INCOMPATIBLE: A strips tags,
                                  so B's "untagged" condition holds for every
                                  sleeper and the return storm arrives.

Tail at 8 threads, grand:

    metric        base        B=1 (earlier)   B=4
    p99.9         1,024 ns    12,288 ns       1,536 ns
    p99.99        2.6 ms       8.4 ms        10.5 ms
    p99.999      83.9 ms      41.9 ms        12.6 ms
    MAX          ~253-333 ms      —          ~73-94 ms

Reproducibility: B4's p99.999 = 12,582,912 ns bit-identical 3/3 (base's
83,886,080 likewise 3/3). The deep-tail improvement is real and stable.

**B=4 is the balance the B=1 note predicted**: deep tail 6.7× better, MAX ~4×
better, p99.9 essentially recovered (1.5× vs 12×), −3 % throughput. A is +10 %
throughput with the grand tail untouched (the predicted no-op — grand holds no
tags) and mixed MAX slightly worse, which is the losing-your-place cost made
visible.

### Status: knobs exist, defaults unchanged — the decision is a judgement call

* **A** is a throughput knob, not a latency knob. +10 % is worth considering on
  its own merits, but it needs soak beyond one benchmark (the mixed-MAX
  degradation says the age ordering does lose information), and it must never
  ship together with B.
* **B=4** is the realtime knob: the only mechanism in this whole sequence that
  cut the deep tail at acceptable cost, with the mechanism verified (returns →
  attempts → tags) rather than inferred. Still 12.6 ms — direction, not
  arrival, for a 1 kHz deadline.
* Anyone enabling either does so per-build, and the latency bench plus the
  128-thread gate are the acceptance tests, as throughout.

### A's tail, measured properly (3 repeats per arm)

    grand:  p99.99 identical (2,621,440 both, 3/3).  p99.999 base 67.1 ms 3/3,
            A 83.9/67.1/83.9 — one histogram bucket, and base itself has sat in
            that upper bucket in earlier sessions, so this is drift across a
            bucket boundary, not an effect.  MAX overlapping (235-328 ms).
    mixed:  p99.9 / p99.99 / p99.999 BIT-IDENTICAL in all 3 repeats
            (1,792 / 3,670,016 / 12,582,912).  MAX overlapping
            (base 68.6-79.8 ms, A 48.5-76.1 ms).

So **A is latency-neutral** — and the earlier single-run note that A worsened
the mixed MAX was noise; retracted. The refined picture: the tags that 38 % of
mixed sleepers held were costing *peer throughput* through `fair_mode_blocks_me`
(hence A's +10 %), but they were never a cause of the tail. A is purely a
throughput knob; B=4 remains the only latency knob.

### B respects privilege (it did not, and the omission was latent)

B as first written returned unconditionally — it did not consult the privilege
stamp. That matters specifically *because of B*: returning leads to an attempt,
an attempt can fail, failing tags, and `tags_total > 0` is exactly the condition
the livelock verdict had been missing. B is what makes privilege reachable, and
it would then have barged straight past the holder.

Measured, the situation has not arisen: with B=4, `priv tries` is still 0.000
(and tagged-list at sleep is only 0.03 — the returned transaction usually
commits rather than sleeping again tagged). So this is not a fix for an observed
defect; it is refusing to let B's correctness rest on privilege happening to stay
dormant.

The guard is free: `_fair_blocks` — `fair_mode_blocks_me` — is already computed
earlier in the same loop iteration, so the condition gains one already-live
boolean.

Verified to be the no-op the reasoning predicts, which is the point of running
it: throughput 8.86 vs 9.06 M/s (within spread) and grand p99.99 / p99.999
bit-identical across repeats. Had it changed anything, the claim that privilege
is dormant would have been wrong.

### Why B costs throughput — it is a scheduling cost, not wasted work

The obvious guess was wasted work: B returns, attempts, and a failed grand
attempt throws away a whole bundle. **Measured, wrong** — the retry rate does
not rise, it falls: attempts/commit over ALL commits is 1.000 under B=4 against
1.019 for base's grand arm. When the returned transaction attempts, it mostly
succeeds.

The cost tracks oversubscription instead (host: `hw.ncpu` 8, 4 performance
cores):

    threads     base        B=4        delta
      4        6.20 M/s   6.12 M/s    −1.3 %   (fits the P-cores)
      8        6.94       6.43        −7 %
     16        7.11       6.35        −11 %
    128        9.68       8.81        −9 %

So: **the backoff is not only a CAS-storm damper, it is a scheduling device.**
A contender that sleeps hands its core to a thread that can make progress. B
stops it sleeping and puts it back on the runqueue, and the aggregate loss is
the CPU time taken from everyone else. With cores to spare the effect nearly
vanishes.

That reframes B's price in a way that matters for the realtime goal. −3 to
−11 % is not an intrinsic cost of the mechanism; it is what it costs *on an
oversubscribed machine*. The standard PREEMPT_RT deployment discussed earlier —
the realtime thread pinned to its own core, so it never competes with the
workers — is precisely the configuration in which B is close to free, and it is
also the configuration in which its latency benefit is wanted. The knob and the
deployment fit each other.

### Choosing the B value — B=1 is dominated; B=4 is the optimum

The earlier B=1 and B=4 figures came from different sessions, which is the
comparison this document keeps warning against. Re-run head to head, grand arm,
3 repeats, medians:

    knob     p99.9        p99.99     p99.999
    base     1,280 ns     3.67 ms    67.1 ms
    B=1     16,384 ns    10.5 ms     41.9 ms
    B=4      2,560 ns    10.5 ms     12.6 ms   (bit-identical 3/3)
    B=8      1,024 ns    12.6 ms     41.9 ms

**B=1 is dominated by B=4** — 6.4× worse at p99.9 *and* 3.3× worse at p99.999,
for the same ~3 % throughput. It returns too eagerly and thrashes. B=8 is the
opposite failure: p99.9 is back to baseline but so is the deep tail, so it buys
nothing. **B=4 sits at the minimum**, and its p99.999 was identical in all three
repeats.

So if B is enabled at all, the value is 4. Whether to enable it is the separate
question, and for KAME today the answer is probably not: base p99.9 is 1,280 ns
against B=4's 2,560, the deep tail nobody currently observes, and KAME's
deadlines are instrument I/O at millisecond scale. B=4 earns its place in a
deployment that has a realtime thread on its own core — where, per the previous
section, its throughput cost also nearly vanishes.
