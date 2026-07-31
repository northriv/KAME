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

## (D) "The oldest returns and tags" — principled, and the tightest bound yet

Two objections to B, both correct: **4 is not a meaningful number**, and the
tail does not cap at ~1 ms the way a return rule should. The second has a clean
explanation — B fires at most *once*. Its condition is
`m_tagged_linkages.empty()`, so the first failed attempt tags the transaction
and B never applies again; escalation resumes. B is a one-shot escape, not a
ceiling.

The principled replacement is the design's own rule: **the oldest contender
returns and tags.** No constant, no timeout — `signed_diff_us_packed(cur, mine)
> 0` is the same oldest-wins comparison `tag_as_contender` already uses. A
sleeper READS the linkage stamp first and acts only if it would win, so the
shared word sees many readers and few writers — the distinction from tagging
unconditionally before every sleep, which made 128 writers and cost 97 %.

    metric            base          B=4            D
    p99.999        67–84 ms      12.6 ms        8.4 ms
    MAX           254–290 ms    63–66 ms     12.5–14.6 ms
    throughput 4t       —         −1.3 %       neutral
    throughput 128t     —         −3 %          −66 %

**MAX collapses 290 → 14 ms, a factor of 20** — the first result in this whole
sequence that looks like a bound rather than a smaller number. And at four
threads, at or under the core count, it is free.

The −66 % at 128 threads is where the "one writer" reasoning breaks: as the
starved transactions age, *many* of them are older than a recently-placed young
tag, so many write and many return; and every time a tag clears, the whole
population sees an empty slot at once. It is a storm again, just a
demographically-driven one.

p99.9 is also unstable under D (14 µs and 2.6 ms in consecutive repeats), which
wants explaining before D could be a default anywhere.

### What this means

D and B are for different deployments, and the split is the same one the
allocator's contract landed on:

* **oversubscribed, throughput-first** (KAME today, 128 threads on 8 cores):
  neither — base is best, and nobody observes the deep tail.
* **cores to spare, latency-first** (the PREEMPT_RT recipe: realtime thread on
  its own core): **D**, free at that thread count and with a 14 ms worst case
  against base's 290 ms.

Which also answers the earlier question about B's throughput cost being a
scheduling cost: so is D's, only more so. Both knobs are cheap exactly where
their benefit is wanted, and expensive exactly where it is not.

### Throttling D with the existing lease — diagnosis confirmed, configuration not

D's cost at 128 threads is a return storm, so the obvious throttle is the one
already there: `PriorityState` carries a per-Linkage `{tid, start_us, lease_us}`
and "a claimant is mid-turn" is exactly what a live lease means. Gated D's
return on it.

    128t throughput   base 9.8 M/s
      D (no lease)          3.6 M/s   (−63 %)
      D + lease (1–10 µs)   3.4 M/s   (−65 %)   no effect
      D + lease (≤100 µs)   3.5 M/s   (−64 %)   no effect
      D + lease (2–5 ms)    5.4 M/s   (−45 %)   helps

**Timescale mismatch**, and the sweep says so cleanly: the lease is designed for
µs-scale spin arbitration (1–10 µs adaptive), while the sleeps it would have to
serialise are ~1,500 µs. By the time a sleeper wakes and looks, the lease has
long expired — a 10 or 100 µs window cannot order events spaced 1.5 ms apart.
Scaling it to the sleep timescale does help, which confirms the diagnosis.

But it is not a usable configuration: at 2–5 ms the lease throttles D's returns
so effectively that it throttles away D's whole point —

    knob            128t        MAX            p99.999
    D                −63 %    12.5–14.6 ms      8.4 ms
    D + lease 2–5ms  −45 %    79.6 ms          16.8 ms
    B=4              −3 %     63–66 ms         12.6 ms

— landing worse than B=4 on *both* axes. Dominated. The lease gate is removed;
plain D stands as the "cores to spare" knob it was measured to be.

The lesson generalises past this knob: the negotiator has two arbitration
timescales — a µs spin/lease layer and a ms sleep layer — and a mechanism aimed
at one cannot govern the other. Every attempt in this sequence that tried to fix
the ms-scale tail with µs-scale machinery (lease here; slot keying and tagging
earlier, both µs-scale writes) failed for a variant of that reason.

### D + A composes — same tail, a third of the cost recovered

A (release tags before sleeping) fought with B: A strips the tag, B's
`empty()` condition then holds for every sleeper, and the return storm cost
−28 %. With D there is no such coupling, and the two compose cleanly.

    128t throughput (5 reps, medians)      grand tail (2 reps)
      D      3.40 M/s  (−63 %)               p99.999 8.4 ms, MAX 11.3–14.7 ms
      D+A    5.47 M/s  (−44 %)               p99.999 8.4 ms, MAX 11.3–14.7 ms

The tail is *identical* — p99.99 6,291,456 and p99.999 8,388,608 in 4 of 4
measurements across both configurations — and a third of D's throughput cost is
gone. **D+A dominates D.** At four threads, both are indistinguishable from base.

The reason A helps here and hurt with B is that A's benefit is independent of
the return rule: a sleeper that has dropped its tag stops blocking peers through
`fair_mode_blocks_me`, which is where A's +10 % came from in the first place. B
*consumed* A's tag-clearing as a trigger; D only reads the tag to decide who is
oldest, so A removes a source of peer blocking without multiplying D's returns.

## Where the knobs stand

    config    4 threads     128 threads    p99.999      MAX
    base          —              —        67–84 ms   254–290 ms
    A          neutral        +10 %       unchanged   unchanged
    B=4        −1.3 %          −3 %        12.6 ms    63–66 ms
    D          neutral        −63 %         8.4 ms    12.5–14.6 ms
    D+A        neutral        −44 %         8.4 ms    11.3–14.7 ms

* **Throughput-first, oversubscribed** (KAME as it ships): **A alone**, or
  nothing. A is +10 % and latency-neutral; the deep tail nobody currently
  observes.
* **Latency-first with cores to spare** (a realtime thread on its own core —
  the PREEMPT_RT recipe): **D+A**. Free at that thread count, and a 14 ms worst
  case against base's 290 ms.
* **B=4** occupies the middle: the only option that improves the tail while
  staying near base throughput *under oversubscription*. If a deployment cannot
  give the realtime work its own core, B=4 is what remains.

All three default OFF; none is proposed as a default here.

### D writes nothing — the tag inside it was redundant (user)

D as first written tagged the linkage before breaking out. Two arguments were
offered for that write, and both were wrong.

*"The tag is how this Tx gets tagged."* No — it is about to attempt, and a
failed attempt tags at the one production site (`operator++`'s pre-commit retry
tag). The in-loop write only made it earlier.

*"Earliness is what stops the other sleepers from returning too."* This was my
hypothesis, and the measurement refuted it. With the write removed:

    entries/rounds   1.57 / 2.00   vs   1.56 / 2.01
    sleeps/commit    1.82          vs   1.82
    slept/commit     3.034 Mns     vs   3.026 Mns
    throughput       within noise at 4 and 128 threads
    grand tail       p99.99 6,291,456 and p99.999 8,388,608 in both

The slot is not maintained by D. It is maintained by the existing tagging
traffic: `operator++` tags on every retry of every multi-nodal Tx, so under
contention the linkage stamp is essentially always non-zero and always carries
the oldest contender's stamp (the CAS is oldest-wins). D's read therefore finds
a live, correctly ordered stamp whether or not D itself ever writes, and the
`_cur == 0` branch — the one that would let everybody return at once — is rare.

Reading the condition also shows D never fires for a Tx that already holds the
tag: `_cur` is then its own stamp, `signed_diff` is 0, not > 0. D fires only
when the slot is empty or held by someone younger. It was always a read.

Keeping it a read is worth more than the CAS and the `m_tagged_linkages` push it
saves. **A policy that cannot mutate negotiation state cannot perturb anyone
else's oldest-wins arbitration**, so D is structurally incapable of the
interaction that made A+B cost 28 %, and A has nothing extra to clear. D is now
purely a "when to stop waiting" rule, and the numbers above are what it costs to
be sure of that: nothing.

## The wait is the wrong thing to bound — knob C is the right one, and its value is derivable

The proposal examined here (user): stop sleeping in fixed 1 ms chunks and give
`cell.wait` an upper bound instead, on the reasoning that every wake-up turns a
thread into a running contender and extra concurrent runners are measurably
expensive.

The physics behind that reasoning is right and is recorded: the spin-admission
sweep measured cap=1 at 965 k/s against cap=2 at 616 k/s, −36 %, and an
independent warm-spinner prototype put one extra full-time runner at 0.81–0.90×.
Two mechanical details, both verified in the source, redirect it.

**The sleeper is not a runner between chunks.** `ReleaseOneCount onedown` is
constructed at `transaction_neg_impl.h:1925`, *before* the `#if
KAME_STM_MIN_RUNNERS != 0` that opens the chunk loop, and lives to the end of the
block. The sleeper is out of `numThreadsRunning()` for the whole round, so chunk
boundaries create no runners; only round boundaries do. Capping the wait below
`ms_actual` therefore multiplies round boundaries and *increases* running-runner
time — the opposite of the intent. (What the chunk loop does pump is *other*
threads: the per-chunk `notify_n_contenders` refill targets
`min_r = hardware_concurrency`.)

**`KAME_STM_MAX_RUNNERS = 2` is already a cap of one.** The gate is
`numThreadsRunning((unsigned)max_r) < (unsigned)max_r` (:1582) and the caller
counts itself, so 2 means "proceed only when no peer is running" and 1 would be
unsatisfiable. The shipping default is already the one-runner setting that the
sweeps favour; there is no 2-vs-1 regression to remove.

**Bounding one wait bounds nothing a caller sees.** The round loop `for(int ms =
0;;)` (:1395) has no iteration limit, and 99.7 % of waits at N=8 already end by
timeout rather than by a wake — verified independently here: every `wake_one()`
and `notify_n_contenders` call site in `kamestm/` is inside `_negotiate_internal`
or its two notify helpers, and `drop_tags_n_privilege` (`transaction.h:1809`)
frees the linkage stamp with a bare CAS to 0 and wakes nobody. The wait is a
*polling period*, not a backstop. Shortening it was already measured as a net
loss (1000→50 µs chunks: 0.94–0.97×), and lengthening it changes a round's
wall-clock not at all, because `t_end = now + ms_actual*1000` is a hard time
budget that an early wake merely restarts.

### The 1 ms chunk is already the derived-correct value

    B = max(KAME_STM_PRIV_PREEMPT_WINDOW_US,
            effective_runners() × KAME_LEASE_US_MAX)

— one arbitration-coherence window, or the time for every servable contender to
take its legitimate lease, whichever is longer. On the 8-core M3 that is
max(1000 µs, 8 × 10 µs) = **1.00 ms**; on a 128-core host 1.28 ms; on Windows
16 ms, which is the timer tick and the floor below which any bound is fiction.
Both terms are existing tuned constants, so the expression tracks them. The
existing constant is not arbitrary and should not change.

### Knob C is the bound, and R follows from the deadline

`KAME_STM_RETURN_CEILING_MS` (:2142, default 0) bounds the *hand-back to the
caller*, which is the quantity a deadline is actually about. It had never been
measured. Measured now, 3 reps, grand arm at 8 threads for the tail and the
3-level mixed test for throughput:

    R        128t thr.   4t thr.   p99.999        MAX (3 reps)         reproducible?
    base        —           —      67–84 ms     183 / 234 / 277 ms      no
    C=3      −58 %          —          —              —                  —
    C=4      −45 %      neutral   12.6 ms (3/3) 15.7 / 16.8 / 18.0 ms   yes
    C=8      −23 %      neutral   41.9 ms (3/3) 44.0 / 45.8 / 51.9 ms   yes
    C=16      −7 %      neutral   58.7–67.1 ms  138 / 150 / 154 ms      yes

The ladder is `ms = max(dt2*mult_wait/10000, ms + 1)` (:1616), so in the +1-ramp
regime the accumulated sleep after R rounds is R(R+1)/2 ms, and one commit makes
`entries/commit` ≈ 1.56 negotiator calls:

    MAX  ≈  R(R+1)/2 × entries_per_commit  [ms]

      R=4  → 10 × 1.56 = 15.6 ms   (measured 15.7–18.0)
      R=8  → 36 × 1.56 = 56   ms   (measured 44.0–51.9)
      R=16 → 136 × 1.56 = 212 ms   (measured 138–154)

Invert it for a deadline D: **R = ceil((sqrt(1 + 8D/e) − 1) / 2)**, e ≈ 1.56.
D = 20 ms → R = 4; D = 50 ms → R = 7; D = 150 ms → R = 13.

This also retires the objection that B=4's "4" was a number without meaning. The
same arithmetic gives it one: R(R+1)/2 = 10 ms is the first rung at or above the
unbounded system's own p99.999 (8.39 ms), so R = 4 is the smallest ceiling that
does not truncate the legitimate distribution.

### Why C is the realtime knob even though it does not measure best

    config    128t     p99.999    MAX (3 reps)              MAX reproducible?
    B=4       −3 %     12.6 ms    54.7 / 55.4 / 224.3 ms    NO — blows out
    D+A      −44 %      8.4 ms    11.3 / 12.4 / 14.7 ms     yes
    C=4      −45 %     12.6 ms    15.7 / 16.8 / 18.0 ms     yes
    C=8      −23 %     41.9 ms    44.0 / 45.8 / 51.9 ms     yes

B=4 is by far the cheapest and D+A has the lowest MAX, so on the raw numbers C is
dominated. **This measurement also corrects B=4's earlier entry**: its MAX was
recorded as 63–66 ms, but a third rep produced 224 ms. That is structural, not
noise — B returns only an *untagged* Tx, so once a Tx acquires a tag it is back
on the unbounded ladder. Its good MAX was a tendency, not a bound.

C is the only knob whose worst case is both reproducible across runs and
computable in closed form before running anything. For a deadline that is the
whole requirement: you need the bound you can *state*, not the one that happens
to measure well. B and D produce an emergent MAX; C produces a designed one.

Composing them does not help — B=4+C=8 gives −24 % and MAX 44.8–67.5 ms, no
better than C=8 alone at −23 %, and B=4+C=16 loses C=16's reproducibility
(59.3 / 62.9 / 115.7 ms).

**Recommendation.** Leave the CV wait at 1 ms and leave chunking alone. Use C,
sized from the deadline by the formula above; C=8 is the reasonable middle on
this host. All knobs stay default OFF.

### The "uncapped numThreadsRunning" win is not one — retracted

Reported above as a free optimisation: the chunk loop calls the **uncapped**
`numThreadsRunning()` once per chunk (:1938), so passing `min_r` as the ceiling
should cut a 128-entry list walk. Implemented and measured; it is not a win, and
the premise was wrong twice over.

**It does not measure faster.** Per-rep interleaved, medians of 3:

    threads   uncapped   ceiling = min_r
      128      9.60 M       9.70 M
       64      9.44 M       9.15 M
        8      6.94 M       6.94 M
        4      6.12 M       6.13 M

**The walk is not a bottleneck at all here.** A semantics-breaking diagnostic
build that forces `ceiling = 2` — pruning the walk to about two entries — is
also indistinguishable from base (128t 9.68 M vs 9.70 M; 64t 9.39 M vs 9.18 M).
There is nothing to reclaim on this host.

**The ceiling cannot prune even in principle.** `effective_min_runners()`
returns `hardware_concurrency()` when `KAME_STM_MIN_RUNNERS = -1` (:643-647), so
`min_r` is the same order as the entry-list length, and under contention the
running sum is 1-2 because sleepers hold `ReleaseOneCount`. The sum therefore
never reaches the ceiling and the full list is walked regardless. On a 128-core
host `min_r` is 128 and the ceiling is exactly the list length.

**The 30 % figure was an already-fixed hotspot.** The only recorded number for
this function is 27.8 %, and `transaction_detail.h:305-308` records it as
*eliminated*: the current per-thread heap-entry design "Replaces the
heap-vector+atomic_shared_ptr design (eliminated the 27.8% hotspot on x86_64
NUMA and the TLS-teardown race)". Reusing that figure as a live cost was a
misreading.

The change is reverted. If per-chunk runner accounting ever does show up in a
profile, the fix is not a ceiling — it would have to be caching the value across
chunks or lowering the refill cadence, both of which change behaviour.

## ScopedWaitBudget — a caller-supplied soft bound on waiting

Implemented (steps 1 and 2 of the four sketched with the user). Not called a
deadline, deliberately: nothing here guarantees a transaction *completes* by the
limit. What it bounds is the **waiting**.

    Transactional::ScopedWaitBudget wb(1000);   // 1 ms for this cycle
    node.iterate_commit_while([&](Transaction<XN> &tr) -> bool { ... });

The API takes a **duration** because that is how a driver thinks, and converts
once at construction to an **absolute** µs limit. Absolute is required, not
stylistic: the negotiator is re-entered ~1.56 times per slow commit and `ms`
restarts at 0 each time, so a relative budget re-armed per entry would bound
nothing. One scope covers every transaction inside it. Nesting takes the tighter
of the two limits — a callee must not be able to grant itself more time than its
caller allowed.

In the negotiator: `t_end` is clamped to the limit, each chunk is clamped to the
remaining budget (via a new `us_override` parameter on `negotiate_sleep`, because
the ms ladder cannot express "300 µs left"), and a loop-tail escape returns at
expiry. All five sites are guarded on the limit being nonzero, and the escape
yields to a privilege holder like every other escape here.

### It works

Grand arm, 8 threads, one budget per commit:

    budget      p99.9      p99.99     p99.999      MAX
    none        1.3 µs     5.24 ms    67.1 ms    275.3 ms
    2000 µs     2.62 ms    2.62 ms     3.67 ms    32.4 ms
    1000 µs     1.31 ms    1.31 ms     8.39 ms    33.3 ms
    500 µs       655 µs     655 µs     1.31 ms    16.6 ms
    200 µs       262 µs     459 µs     2.10 ms    11.0 ms
    100 µs       164 µs     229 µs      459 µs     1.75 ms

The percentiles track the budget (500 µs → p99.9 at the 655 µs bucket), and the
100 µs budget cuts the worst case 156×. All nine STM tests pass.

The residual is honest: a 100 µs budget still shows a 1.75 ms max. That is not
waiting — it is the **unbounded retry count** after the budget expires, plus OS
jitter, plus the cases where a privilege holder blocks the escape. Bounding that
is step 3 (claim privilege on exhaustion so the attempt that follows wins), which
is what turns "stops waiting on time" into "overshoots by one attempt".

### Two rejected implementations, both caught by measurement

**`Snapshot::m_wait_limit`, filled by the ctors that stamp `m_started_time`.** The
reasoning was that the negotiator should read a member rather than TLS. Measured
−1.9 % at 8 threads and −0.9 % at 4 (7 interleaved reps; lower in 6 of 7 at 4
threads). The ctors are the hot path and the negotiator is not — and the member
bought nothing anyway, because the value was already hoisted into a local once
per negotiator call, so the sleep loop never touched TLS in either version.

**A separate `XThreadLocal<int64_t>` read once per negotiator call.** Cleaner, and
still wrong: **−2.36 % at 8 threads, lower in 7 of 7.** `XThreadLocal<T>::operator*`
caches its pointer in a function-local `thread_local` that is distinct per
instantiation, so a second `XThreadLocal` of a different type is a second TLS
wrapper call — on macOS a `_tlv_get_addr`, the same call that cost 4 % on the pool
allocator's cross-thread free path earlier in this programme.

**What shipped**: priority and wait limit share one slot (`detail::TxContext`), so
`_negotiate_internal`'s existing single read serves both and the budget costs one
extra load from a cache line it already has. Re-measured with the same
same-source two-build-dir A/B:

    threads   OFF        ON         delta     ON lower in
       8      6.895 M    6.914 M    +0.27 %      6 / 11
       4      6.106 M    6.099 M    −0.11 %       3 / 7
     128      9.396 M    9.666 M    +2.88 %       1 / 7

No measurable cost when unused. `KAME_STM_WAIT_BUDGET` (default 1) compiles the
whole thing out; with 0 the API is not declared, so a caller expecting a budget
gets a compile error rather than a silent no-op.

The general lesson, third time in this programme: **on macOS, "just one more
thread-local read" is a TLS-wrapper call, not a load.** Put the new field in a
slot something already reads.

## HIGHEST is not a priority — it is an exemption from politeness

Asked whether HIGHEST can claim privilege, and whether a thread that does not
hold privilege can be yielded to at all.

**Only privilege buys deference.** In the default per-Linkage mode
`fair_mode_blocks_me` returns false immediately unless the linkage slot carries
someone else's unexpired **Reserved** stamp — there is no priority term. The one
other form of yielding, a younger transaction sleeping for an older tag, is
priority-blind too. So a HIGHEST transaction that has not claimed privilege gets
exactly zero deference from anyone.

**HIGHEST's actual special case is that it never negotiates.**
`if(entry_pr == Priority::HIGHEST) break;` is the first statement of the round
loop (`transaction_neg_impl.h:1429`). It never spins and never sleeps.

**The privilege claim itself is priority-blind.** The claim block sits *before*
that loop, so HIGHEST does pass through it — but `#if KAME_PER_LINKAGE_PRIVILEGE`
(default) opens with a literal `(void)entry_pr;` (`:1306`), and the verdict
threshold is `clamp(sig_C*2, 3, hw_procs)` (`:369`), not the per-priority one.
The per-priority thresholds from `priority_probe_info()` — HIGHEST 2, NORMAL 3,
the rest 4 — are **dead**: `pinfo` is consumed only as `pinfo.name` in a
diagnostic printf (`:389`).

### Measured, per priority (grand arm, 8 threads, 4 s, NEG_DIAG)

An earlier aggregate reading here said HIGHEST claims privilege at 0.381 grants
per slow commit. **That was wrong** — the aggregate cannot attribute a grant to a
thread, and adding a HIGHEST thread raises everyone else's retry count. Split by
priority:

    -P 1        slow n   rounds/cmt  sleeps/cmt   priv grants
      HIGHEST        5        0.20        0.00        0.000
      NORMAL      1869        4.88        8.77        0.541

    -P 2        slow n   rounds/cmt  sleeps/cmt   priv grants
      HIGHEST     1334        2.71        0.00        0.988
      NORMAL      5119        3.95        2.78        0.231

With one HIGHEST thread it claims privilege **never**, and does not need to: it
has 5 slow commits in four seconds against the NORMAL group's 1869, because the
NORMAL threads are asleep (8.77 sleeps/commit) and the CAS field is clear.

With two, that collapses. The two impolite threads collide, HIGHEST slow commits
go 5 → 1334, and only then does the livelock verdict fire (0.988 grants ≈ one per
slow commit) — the reactive, late mechanism, reached only after starvation.

**So HIGHEST is a positional advantage that exists only while exactly one thread
takes it.** That is the same curve as the earlier -P sweep: 1 HIGHEST costs 4 %,
4 cost 10×, 8 cost 42×.

### What this means for the wait budget

There is no proactive route to being yielded to. Privilege is granted only to a
transaction that is *already starving* and still owns every tag it took. For a
thread that has declared a wait budget, that is exactly backwards — it wants
deference *before* it misses, not after.

This is what step 3 has to supply, and it confirms the user's constraint that it
must not apply to NORMAL generally: if every thread could claim privilege on
budget expiry, a deployment where budgets are common converges on the -P 8 case
(0.06 M/s). The host for step 3 is "the few threads that declared a budget", not
a priority level — and notably **not** HIGHEST, for which a wait budget is inert
because HIGHEST never waits.

`transaction_latency_bench` gains `-P N` (first N threads at HIGHEST) and, under
NEG_DIAG, a per-priority split of the negotiator counters.

## Closing the wait-budget holes, and what the budget can honestly promise

The first cut of the budget escape sat only at the loop tail. Measured against
a 1 µs budget it produced **14.23 rounds/commit and 0.18 sleeps/commit** — it was
being bypassed. Three fixes, all pure logic:

* **The check moved to the top of the round loop.** That is the only point every
  path through a round passes; the fair-spin `continue` (`:1867`) skips the tail
  entirely.
* **The fair spin is clamped to the budget.** A 2 ms busy-spin is longer than
  most budgets; it now ends at whichever of the two comes first.
* **Expiry no longer yields to a privilege holder.** Gating the escape on
  `fair_mode_blocks_me` meant a budget could be exceeded without limit whenever a
  peer held privilege. Returning is not barging — the caller attempts its CAS and
  loses to a committing holder like any other loser. What we decline to do is
  sleep.

Result, grand arm at 8 threads: with a 100 µs budget, **p99.999 262 µs and MAX
559 µs**, against 83.9 ms and 257.6 ms unbudgeted.

### A rejected fix: an OS-derived minimum sleep

The diagnostic that found the remaining sleeps was one line:

    REQUESTED vs ACTUAL sleep: asked 1000 ns/sleep, got 209530 ns/sleep (209.53x)

The clamp was working — the negotiator asked for 1 µs — and the OS returned in
210 µs. Measured across requests on this host (Apple M-series, `__ulock_wait`):

    asked    1 µs -> 159 µs (159x)      asked   99 µs -> 113 µs (1.14x)
    asked    4 µs ->  34 µs (8.4x)      asked  199 µs -> 224 µs (1.13x)
    asked   18 µs ->  45 µs (2.45x)     asked  998 µs -> 1113 µs (1.11x)
    asked   48 µs ->  90 µs (1.87x)

The obvious fix — a `KAME_NEG_MIN_SLEEP_US` floor below which we skip the wait —
was **rejected by the user, correctly**: that number is undocumented platform
behaviour, not a contract. It moves with the OS version, the hardware, the QoS
class and the power state, and baking it into the general non-realtime path
makes every KAME build depend on it. The library bounds what it *chooses* to
wait; the platform's wake latency is the platform's, and the caller's margin
covers it. The table above is documentation, not a constant.

### The regression test, and why it asserts p99.99 rather than the max

`transaction_wait_budget_test` (ctest) runs the contended grand pattern with a
budget on every commit and fails if latency exceeds `budget + slack`.

The first version asserted on the **max** and failed about half the time. The
attempt count says why: the commit that produced the max had **one or two
attempts** — including the unbudgeted 220 ms and 411 ms maxima, which had
exactly one. One attempt means no retry storm, and with an absolute budget
shared across the commit's negotiator entries it means no overlong wait either.
What is left is the OS not scheduling the thread. A max-based assertion tests the
platform scheduler, not this library.

At p99.99 the budget is tracked almost exactly, and the overshoot is a **fixed
~200 µs independent of budget size**:

    budget   100 µs -> p99.99  150 / 165 / 214 / 263 µs
    budget  1000 µs -> p99.99  1157 / 1163 / 1174 µs
    budget 10000 µs -> p99.99  10128 / 10129 / 10129 µs

So the slack defaults to **1 ms** — about five times the overshoot the mechanism
actually produces, and still below the unbudgeted p99.99 of 2–5 ms, which is what
gives the assertion power. Each arm prints `(no power)` when the control run did
not itself breach that arm's limit; a 10 ms slack marks every arm no-power on
this host, i.e. makes the test unfalsifiable. Overridable via
`KAME_WB_TEST_SLACK_US` for slow or loaded machines. The max is still printed
with its attempt count, because that pair is what separates the two causes.

Non-regression re-checked after all of this: 8 threads, 11 interleaved reps, no
budget set — **−1.22 %, ON lower in 5 of 11**. Coin flip. All ten tests pass.

### Is step 3 still needed?

Less clearly than before. The user's observation — that not going to CV on expiry
is simpler than granting privilege — is supported: one HIGHEST thread (which is
exactly "never waits") measured 5 slow commits in four seconds without ever
claiming privilege. If only a few threads carry budgets, step 2 alone may be
enough. Step 3 stays unbuilt until a workload shows it is needed.

## The four knobs are gone; the measurements stay

A, B, C and D are removed from the code. Every finding above stands — this
section records why none of them was worth carrying as a `#if`.

**They were untested.** Nothing in `kamestm/tests/` or `.github/` ever built
with one enabled. Four preprocessor branches in the hottest function in the
library that no build compiles will rot, and the reproducibility that made C
attractive would rot with them.

**A (`CLEAR_TAGS_BEFORE_SLEEP`) was actively harmful, which the earlier
measurements did not show.** It calls `snap.drop_tags_n_privilege()`, emptying
`m_tagged_linkages` — and the livelock verdict's `tags_total` *is*
`snap.m_tagged_linkages.size()`. So after A fires, the next negotiator entry
cannot satisfy `tags_total > 0` and privilege can never be claimed. A trades the
only escape from starvation for +10 % at 128 threads, having measured no benefit
at 4. It also explains A+B's −28 %: B's condition is `empty()`, which A makes
permanently true, so every sleeper returns at once. The constraint is now a
comment at the sleep site.

**B (`UNTAGGED_RETURN_MS`) never had a bound.** It returns only an *untagged*
Tx, so a Tx that has acquired a tag is back on the unbounded ladder — the
54.7 / 55.4 / 224.3 ms MAX triple. `ScopedWaitBudget` bounds the wait without
that hole and per caller.

**C (`RETURN_CEILING_MS`) is the one with a real claim** — the only knob whose
worst case is both reproducible (3/3 within a few ms) and computable in closed
form before running anything. It loses on shape: the cost falls on every thread
(−23 % for a 44–52 ms MAX at R=8) whereas the budget costs −4.9 % and gets
MAX 608 µs on the one thread that asked. It remains the right answer for a
deployment that cannot add call sites; that deployment does not exist yet, and
the formula above reconstructs it in about five lines.

**D (`OLDEST_RETURNS`) is dominated.** Best tail of the four (MAX 11.3–14.7 ms)
but −44 % even with A, against the budget's 608 µs at −4.9 %. Its p99.9 also
swung between 14 µs and 2.6 ms across repeats and was never explained.

Removing all four left the shipping binary **byte-identical**
(`663136d0…` before and after), which is the proof that they were dead in the
default build and that nothing else came out with them.

What remains in the negotiator is one realtime affordance,
`ScopedWaitBudget` — tested in ctest, default on, and measured free when unused.

## Why the sleeper is invisible: the ctor tags *after* it negotiates

The starvation chain diagnosed earlier ends at `tags_total == 0`. This is the
mechanism, and it is an ordering inversion rather than a missing call.

`ScopedNegotiateLinkage`'s constructor does, in this order
(`transaction_negotiation.h`):

    :404    _negotiate();                     // or _negotiate_after_retry_pause
            ...                               // 83 lines
    :487    m_snap->tag_as_contender(m_link);

`_negotiate()` is what reaches `_negotiate_internal()` and therefore
`negotiate_sleep()`. So **the CV wait always happens before the tag**. This holds
for all twelve construction sites and for both `TagMode`s — `OnEntry` vs
`OnExit` only selects *which* of ctor/dtor tags, not whether the tag precedes the
ctor's own negotiation.

The destructor gets it right, and says so:

    // Tag is performed BEFORE the wait below so that any subsequent
    // notify_n_contenders walking tid_bitset can find us and wake our
    // sleep slot.

So the rule is stated in the code and followed in one of the two places.

### Measured (grand/mixed/leaf arms, 8 threads, 3 s)

    arm     tagged-list size at sleep    sleeps holding >= 1 tag
    leaf    — (never sleeps: uncontended)
    grand   0.00                         0.0 %
    mixed   1.05                         39.9 %

Grand-scope is 100 % untagged at sleep. Mixed is roughly 60 % untagged, and the
tags that *are* present belong to **other** linkages the same transaction already
walked (a multi-nodal commit visits several, and `Transaction::operator++` tags on
retry) — not to the linkage about to be slept on. The invariant is therefore:

  **a thread is never tagged on the linkage it is about to sleep on.**

That is exactly why `notify_n_contenders`, which walks `tid_bitset` for the
linkage it is waking, cannot find the sleeper, and why the livelock verdict's
`tags_total > 0` was unsatisfiable in the grand arm.

### Not a defect to fix — it is the −97 % result, already on record above

**Correction to how this section was first written.** It framed "the ctor tags
after it negotiates" as an ordering defect awaiting a two-line fix. That fix is
§"Candidate *just place the tag*" earlier in this file: the tag was placed
unconditionally immediately before entering the sleep path — the site that *is*
reached — and throughput at 128 threads went **9.6 M/s → 0.32 M/s, −97 %**.
Reverted. The ordering is a *consequence* of the throughput constraint, not an
oversight, and the same applies to the intuitive "the sleeper should register
itself": sleeper-registry broadcast cost −38 %, wake-one −8 % with a worse tail.

The reason is structural and stated with the −97 % result: the sleep cell is a
shared **userspace** atomic, not a kernel futex queue. Spreading waiters across
512 cells by tid is load-bearing for throughput, so any scheme that makes a
waiter findable *by the contended address* concentrates waiters onto that
address's cell — which is the cost. Findable implies concentrated implies slow;
spread implies fast implies unfindable.

What this section does add is the second half of the picture, which the earlier
entries did not spell out: the wake machinery can only ever target

  * previous successful **owners** — `observe()` has one call site
    (`transaction_neg_impl.h:1394`) and it records `loadPriority().tid`, written
    by `tags_successful_cas()`, i.e. by whoever's CAS succeeded;
  * the current **blocker** — the chunk loop's direct `stamp_tid` wake;
  * the **privileged** TID — woken unconditionally, outside any bitset.

Both passes of `notify_n_contenders` iterate only the waker's bitset; there is no
catch-all sweep of the 512 slots. So a thread that has never yet committed on a
linkage belongs to none of the three sets, and its only exit from `cv.wait` is
the timeout. That is not a missing bit to set — a waiter has no bit that
identifies *itself* anywhere, by design.

The one direction not yet costed remains the one named with the −97 % result: a
per-linkage **list** of waiter slots, keeping waiters spread across cells while
the Linkage carries the state to enumerate them. That adds state to `Linkage`, a
hot structure, and should be costed before it is attempted.

### Correction recorded

A first pass at this concluded that `OnEntry` sites with `retry != 0` never tag
at all, from the dtor condition `!(m_eager && m_should_tag)` plus an apparent
absence of any tag in the constructor. The constructor does tag — the call is at
the end of a long ctor body, in each of three overloads, and the first reading
covered only the first overload's opening lines. The dtor comment claiming "ctor
already tagged" is accurate. The defect is the ordering, not a missing tag.

## The uncosted direction, costed: waking the waiter is neutral and useless

The section above named one direction still worth trying — publish the waiter's
identity so a committer can wake it, without touching either the arbitration
stamp (that was the −97 %) or `m_tid_bitset` (whose popcount is `sig_C`, feeding
`retry_thresh_dyn = sig_C*2`, the adaptive lease growth, and
effective_min/max_runners; inflating it moves every consumer the wrong way).

Implemented as the cheapest possible form, mirroring the one wake that already
bypasses the bitset (the chunk loop's `stamp_tid` → slot → `wake_one()`):

* `Linkage::m_waiter_tid`, one 16-bit word.  One **relaxed store** of the
  sleeper's tid immediately before the sleep section — no tag, no bitset, no RMW.
* A wake from `Linkage::tags_successful_cas()`, i.e. **on release** — the wake
  the system never had: `drop_tags_n_privilege` zero-stores the stamp and
  notifies nobody, and `_negotiate()` returns before `_negotiate_internal` (which
  owns every wake site) once the stamp is clear.  Load, `exchange(0)` only when
  somebody is asleep, `wake_one()`.

### The mechanism fires, abundantly

    WAITER: published 2.82/commit, woken by a committer 3.17/commit
            (112.6 % of publishes)

Waiters are found and woken. And:

    metric                OFF        ON
    sleeps/commit         14.21      17.49
    slept/commit          23.9 ms    27.0 ms
    rounds/commit          3.81       4.01
    priv grants            0.000      0.000
    asked/got per sleep    1.12x      1.03x

    throughput 128t         —        −0.8 %
    throughput 8t           —        −0.3 %
    grand p99.99        2 621 440  2 621 440   (identical)
    grand p99.999      83 886 080 83 886 080   (identical)
    grand MAX             332 ms     330 ms

It wakes, the gate refuses, it sleeps again — so the sleep *count* rises while
the total sleep time rises with it, and the tail does not move by one bucket.

### Why this is the decisive one

Every earlier attempt in this line was confounded by its own cost: at −97 %,
−38 % or −8 % a null tail result could always be blamed on the damage rather than
on the hypothesis. This one costs 0.3–0.8 % — it touches neither the arbitration
nor the contention estimate — and is *still* null. So the wake is not the binding
constraint; the sleeper is refused at the **admission gate**, exactly as the
56,000 system-wide commits completing during one slow commit's wait already
suggested.

That also closes the per-linkage waiter-list design named above as the one
uncosted option. Its only advantage over this experiment is waking *more*
waiters, and waking one produced no tail movement while adding sleeps and rounds.
There is no version of "make the waiter findable" left to try.

Reverted, as A/B/C/D were, and for the same reason: an untested `#if` in the
hottest function in the library will rot, and the value is in the measurement.

**What remains is unchanged and is not inside the negotiator**: fewer contenders
per linkage (commit at a narrower scope) and less per-commit work (bundle churn
is O(subtree)). A bounded commit is a contract precondition on the caller.

## "Take privilege away from a slow below-NORMAL Tx" — already there, and unreachable

The lever exists and is coherent. `stamp_is_expired_lowprio` treats a
lowprio-tagged privilege stamp older than
`min_privilege_age_us(SCRIPTING) + KAME_STM_PRIV_MAX_HOLD_US` = 1 ms + 50 ms =
**51 ms** as expired, and all five consumers agree, which is what stops a
Reserved stamp going stuck:

    :165  try_register_privileged_tidstamp  a challenger may take it over
    :223  i_am_privileged_now               the holder stops believing it has it
    :235  i_am_privileged_now
    :295  fair_mode_blocks_me               peers stop yielding to it
    :313  fair_mode_blocks_me

It is never reached. Grand arm, 8 threads, 4 s, with `-L N` putting N threads at
`Priority::SCRIPTING`:

    SCRIPTING 1 / NORMAL 7    both groups: priv tries 0.000  grants 0.000
    SCRIPTING 2 / NORMAL 6    both groups: priv tries 0.000  grants 0.000
    SCRIPTING 4 / NORMAL 4    both groups: priv tries 0.000  grants 0.000

Nobody claims privilege at any mix, so there is nothing to evict. The claim needs
the livelock verdict — `tags_total > 0 && tags_owned == tags_total &&
retries >= clamp(sig_C*2, 3, hw_procs)` — and `tags_total` is 0 at sleep. The
eviction sits downstream of a gate that does not open.

### But the bench is symmetric and the real case is not

Every thread here does the same whole-tree commit. In KAME the below-NORMAL work
is shaped differently: a UI redraw or a Python/MCP session takes a **Snapshot of
an ancestor** (often the measurement root) while drivers commit their own
subtrees. That asymmetry is the documented bundling collision — an ancestor
snapshot absorbs the target's packet — and it is precisely the case where a
SCRIPTING thread could plausibly hold privilege long enough for the 51 ms
eviction to matter. This bench cannot produce it.

So the question "does a below-NORMAL thread ever hold privilege in KAME" is open,
and only the application answers it. The instrumentation is now in place:
`XPrimaryDriver`'s record-commit counters on the acquisition side, and
`[ll-probe]` under `KAME_STM_PRIV_DIAG` for the verdict itself. If those show
grants in a real session, the 51 ms figure becomes worth tuning; until then it is
a correct mechanism with no observed trigger.

`transaction_latency_bench` gains `-L N` alongside `-P N`.

### Should the 51 ms be capped by the wait budget?

Right instinct, and it is already satisfied — in the waiter, not in the predicate.

A budget-carrying thread never waits 51 ms for a lowprio holder's privilege to
expire, because the budget escape is **deliberately ungated** on
`fair_mode_blocks_me` (`transaction_neg_impl.h:1464-1470`): "an expired budget
must stop waiting even while a peer holds privilege, or the budget is not a bound
at all." Whatever the eviction deadline is, the budget thread has already left.

Putting the cap inside `stamp_is_expired_lowprio` would break the invariant that
makes the mechanism safe. That predicate is documented as a single source of
truth for three consumers that must agree, and their positions differ:

    try_register_privileged_tidstamp   challenger    its own budget is meaningful
    i_am_privileged_now                HOLDER        has no wait to bound
    fair_mode_blocks_me                yielding peer its own budget is meaningful

Making it caller-dependent puts the holder and its peers in structural
disagreement — a budget peer would read "expired" while the holder still reads
"valid" — and that disagreement is precisely what the comment says makes a
per-Linkage Reserved stamp go stuck. The predicate is shared by a party that has
no waiting to bound, so a waiting-derived cap cannot live there.

What 51 ms still governs for a budget thread is only whether it may *take over* a
stale holder's privilege, which it does not need. For a thread **without** a
budget the full 51 ms applies; whether that is too long is a tuning question that
cannot be answered while grants are 0.000 in every configuration measured.

## A revocable priority must be given a way to fail

The rule (user): **a priority that can have its privilege taken away must have a
timeout.** Revocability without a failure path is not fairness — the thread keeps
retrying with no protection and no exit. The revocable set is exactly what
`stamp_is_expired_lowprio` acts on, i.e. `lowprio_mask_for_current_priority()`
(`transaction.h:517-524`): **LOWEST, UI_DEFERRABLE, SCRIPTING**. NORMAL and
HIGHEST are excluded by the same symmetry — their privilege never expires, so
they are never revoked, and losing a driver record to STM contention is a
semantic no driver expects.

That is a better rule than the per-level reasoning it replaced (which weighed
"a frozen GUI is worse than a failed .kam load"), because it is derived rather
than judged.

The risk is not theoretical and this programme increased it: a HIGHEST
acquisition loop never negotiates and a budget-carrying thread stops waiting, so
slow below-NORMAL work on a node they touch can be retried indefinitely. The
only pre-existing exit was the HANG watchdog `abort()`ing the whole process after
3 x 5 s.

### The throw is a host-installed hook, which is why it is one line in KAME

The first cut threw a new `StarvationTimeoutError` unconditionally. That was a
crash risk, caught by asking whether KAME catches it — it does not. KAME catches
`XKameError` at its connector boundaries (`kame/xnodeconnector.cpp`, six sites)
and nowhere catches `std::runtime_error`; there is no `QApplication::notify`
override and no try/catch around `app.exec()` / `processEvents()`; and
`main.cpp:220` puts the whole GUI thread at UI_DEFERRABLE, squarely in the
revocable set. A new type escaping a Qt slot terminates the process, which is a
worse outcome than the freeze the bound prevents.

Covering a new type meant catches at **seven** UI_DEFERRABLE thread entry points
— `main.cpp:220`, `graphntoolbox.cpp:122`, `xpythonmodule.cpp:347`,
`xpythonsupport.cpp:149`, `xrubysupport.cpp:179` and `:317`,
`xscriptingthread.cpp:108` — plus the six connector chains, and anything missed
is fatal.

So kamestm calls a hook instead:

    using StarvationHandler = void (*)(unsigned retries, long long age_us);
    setStarvationHandler(h);

**No handler is the default and means no throw** — the transaction keeps
retrying exactly as before, so enabling the bound cannot by itself introduce an
unhandled exception. KAME installs one handler in `main.cpp` that throws
`XKameError`, and every catch site it already has works unchanged. Starvation
becomes an ordinary reported KAME error on the same footing, and with the same
coverage, as every other `XKameError` — no new unhandled class. A handler that
returns instead of throwing is also allowed, for hosts that want to count and log
and let the retry continue.

The bound itself is **1000 ms**, which has provenance rather than being
invented: the Priority enum's original doc-comment promised SCRIPTING
"yields to *everything* for the first second of any contention, then claims
privilege so the request still eventually completes". Privilege never fires
(grants measured 0.000 in every configuration), so the promise was never kept.
This keeps it by the other route — "then gives up cleanly" instead of "then
claims privilege". `StarvationTimeoutError` derives from `std::runtime_error`, so
pybind11 hands the SCRIPTING caller a clean Python exception.

    default 1000 ms, 4 SCRIPTING threads, grand arm, 5 s : 0 firings
    fast-path cost, 8 threads, 11 interleaved reps       : +0.78 %, lower 4/11
    128 threads / 4 threads                              : +2.08 % / +0.82 %

So it neither hair-triggers nor costs anything measurable.

### One lowprio thread does not starve; two do

Measured with the bench at a 2 ms bound, grand arm, 8 threads:

    -L 1   does not fire        -L 4   fires
    -L 2   fires                -L 8   fires

A lone lowprio thread gets through. Lowprio threads starve **each other** —
they are excluded from the per-Linkage owner-skip lease (`_neg_apply_lease`) and,
for LOWEST, from the jittered gate, so neither can inherit its way past the
other.

**KAME is already in that regime.** It runs three UI_DEFERRABLE threads: the
main/GUI thread (`main.cpp:220`), the graph toolbox
(`graphntoolbox.cpp:122`) and the Python interpreter
(`xpythonsupport.cpp:149`). And `Priority::LOWEST` is set nowhere in the tree, so
the revocable set in practice is UI_DEFERRABLE (three threads) plus SCRIPTING
(MCP/AI, opt-in behind the sticky trapdoor).

### The test pins the mechanism, not the contention

`transaction_starvation_test` drives `iterate_commit_if`'s retry path directly —
returning false retries unconditionally, so one thread ages one transaction past
the bound with no contention at all — and asserts all five priorities plus the
retry-gate case. Deterministic: 3/3 runs, 6/6 cases.

Manufacturing real starvation was tried first and is not usable as a ctest. It
was flaky in both directions: a two-level tree never starved where the bench's
three-level one did (bundle churn is O(subtree), and the intermediate level is
what makes a root commit heavy enough), and once the starved *peers* caught their
own exceptions and restarted, the victim stopped starving too. Contention
dynamics are what the bench is for.

### There is no "time for HIGHEST to take privilege back"

Asked whether the budget should set it. It cannot, because HIGHEST never concedes
privilege in the first place.

`_negotiate_after_retry_pause` (`transaction_neg_impl.h:697-703`) does route a
thread into negotiation when a peer holds privilege, even at `retry == 0`:

    if(retry == 0 && !NC::fair_mode_blocks_me(...)) [[likely]] return;
    retry_pause(retry);
    _negotiate();

But `_negotiate()` reaches `_negotiate_internal()`, whose round loop opens with
`if(entry_pr == Priority::HIGHEST) break;`, so HIGHEST returns without spinning
or sleeping — it pays `retry_pause(0)`'s CPU relax and nothing else. Measured: one
HIGHEST thread among seven NORMAL shows **sleeps/commit 0.00** and 5 slow commits
in four seconds.

So the times that exist, and who they bind:

    party      waits for a privilege holder      budget-settable?
    HIGHEST    never                             no — a budget is inert on it
    NORMAL     51 ms (stamp_is_expired_lowprio)  not in the predicate (the
    lowprio    51 ms                             three-way agreement), but a
                                                 budget-carrying thread leaves
                                                 earlier via its own ungated
                                                 escape

"How long until I stop deferring to a privilege holder" **is** budget-settable —
for the threads that defer. HIGHEST is not one of them, and that immunity is
exactly why it does not scale past one such thread (the -P sweep: 1 costs 4 %,
4 cost 10x, 8 cost 42x) and why `AcquisitionPriority` is scoped to the
acquisition loop rather than the thread.

### Where the check belongs: two sites, not four

First placed in `iterate_commit` / `_if` / `_while`. Wrong on both counts.

**Too many sites, and it missed Python entirely.** The commit retry step is
`Transaction::operator++`; all three `iterate_commit` variants reach it through
`for(...;;++tr)`, and Python's `Transaction.__next__` reaches it through
`commitOrNext()`, which calls `++(*this)` when the commit fails. So one site
replaces four — and the fourth was the one that mattered, because the Python
retry loop lives in the binding, not in `iterate_commit`, so the priorities most
likely to starve (the interpreter thread is UI_DEFERRABLE, and a script may raise
itself to SCRIPTING) were the only ones with no bound at all.

**And it missed Snapshots, which is arguably the more important path.** A
`Snapshot` is read-only and has no `operator++`; its retry loop is
`for(int retry = 0;; ++retry)` inside `Node::snapshot()`
(`transaction_impl.h:2125`), unbounded. That is the path a graph redraw takes when
it snapshots an ancestor — the GUI-freeze case that motivated the bound in the
first place.

So `throw_if_starved_` now takes a `Snapshot` (both fields it reads live on the
base) and is called from exactly those two places. `Node::snapshot()`'s loop body
runs on *every* snapshot, so this is now a hot path; the retry-count gate keeps it
to one integer compare there, and it measures free:

    threads   OFF        ON         delta      ON lower in
       8      6.772 M    6.809 M    +0.54 %       2 / 9
       4      5.990 M    6.005 M    +0.26 %       2 / 5
     128      9.423 M    9.337 M    −0.91 %       3 / 5
    leaf p50  192 ns     192 ns     identical

`transaction_starvation_test` covers the commit side deterministically (it drives
`iterate_commit_if`'s retry path). **The Snapshot side is not covered by a test**:
that loop only retries on a genuinely DISTURBED CAS, which cannot be forced on
demand, and the same shared helper is what both call. Worth stating rather than
implying the coverage is complete.

### Probing it from inside KAME — attempted, then dropped

A `kame/script/starvation_probe.py` existed briefly. Its first version
manufactured real contention — several threads at a revocable priority committing
at whole-tree scope for ten seconds — and was replaced (user) with a **single slow
transaction**: deterministic, no other threads, no drivers, a dozen writes instead
of thousands. Then the script itself was dropped (user: not needed). What it taught
is kept here, because both points are properties of the bound rather than of the
script, and the second one is a trap for anyone driving the STM from Python:

* **Slow is not enough.** The bound needs an age past the limit *and* at least
  `KAME_STM_LOWPRIO_STARVE_MIN_RETRIES` (8) retries — the gate is what keeps the
  clock off the fast path. A transaction that merely takes two seconds has a retry
  count of 0 and does not fire.
* **In Python the retry count only advances on a FAILED commit.**
  `Transaction.__next__` calls `commitOrNext()` only when the transaction was
  modified, and `commitOrNext()` reaches `++(*this)` — the increment, and the
  bound — only when that commit fails. A body that just sleeps loops without
  incrementing anything, because Python has no `iterate_commit_if` whose
  `continue` would drive `++tr`.

So reaching the bound from outside `iterate_commit_if` means modifying the target
and then committing a **nested** transaction on the same node, invalidating
itself. `transaction_starvation_test` pins that (measured: 5 retries,
deterministic over 3 runs) and keeps doing so now that the script is gone — no
Python-side test can run on a host without the Qt build, so this is the only place
the fact is checked rather than merely asserted.

The consequence worth remembering: **a Python transaction that is merely slow can
never hit the starvation bound**, however long it runs, because nothing increments
its retry count. The bound protects against being *starved by others*, not against
being slow on your own.

## "grants 0.000" is a frequency measurement, not a verdict on necessity

This file says several times that the privilege claim never fires — grants
measured 0.000 in every configuration, across arms, thread counts and
NORMAL/SCRIPTING mixes. That is accurate and it is easy to misread, as I did:
I proposed that the per-linkage privilege machinery was therefore dead weight and
a removal candidate. **That is wrong, and it would break a verified property.**

`kamestm/tests/VERIFICATION.md:317` is explicit: *"The TLA+ priority mechanism
mirrors the per-linkage privilege path in `transaction.h`
(`KAME_PER_LINKAGE_PRIVILEGE=1`, the default)"*, with a symbol-by-symbol
correspondence —

    TLA+                 C++
    priorityTag[n]       Linkage::m_transaction_started_time
    MyTag(t)             Snapshot::m_started_time
    TagAfterFail         Snapshot::tag_as_contender(link)
    CanProceed           i_am_privileged_now / fair_mode_blocks_me
    PreemptTag           the preempt window inside tag_as_contender
    ClearMyTags          drop_tags_n_privilege()

— and `Privilege = TRUE` is set in the model-checked configs, the liveness one
included. The machinery **is** the implementation of the verified
livelock-freedom mechanism. Removing it severs the spec-to-code correspondence
and deletes the mechanism whose absence TLC says produces livelock.

The error has a name worth remembering: **for a livelock-freedom mechanism, the
cases that matter are exactly the rare adversarial interleavings that measurement
does not reach.** Never observing it fire at eight threads says something about
frequency and nothing about necessity — covering what measurement cannot is what
the model checking is for. Treating an absence of observations as an absence of
need inverts the relationship between the two.

### Two guarantees, two mechanisms, and they are not substitutes

    property                        mechanism                     verified by
    bounded waiting / failure       ScopedWaitBudget, the          measurement
                                    starvation timeout             (ctest, bench)
    livelock freedom (progress)     per-linkage privilege          TLA+ TLC

So a NORMAL transaction carrying no budget is not unguaranteed — it has
**progress**, just not a *time* bound. The earlier framing ("NORMAL with no budget
is still unbounded") was about time and is correct about time; it silently implied
there was no guarantee at all, which is not.

That also settles the three-tier design's relationship to privilege. The tiers do
not *depend* on it for their bounds — HIGHEST never waits, NORMAL's escape is the
budget and is deliberately ungated on `fair_mode_blocks_me`, and the lowprio
timeout reads only age and retry count. But privilege is the separate pillar that
makes progress hold at all. Independent, not redundant.

## HIGHEST's invariant is broken by the standard secondary-driver pattern

The three-tier design rests HIGHEST on a deployment invariant: **realtime threads
must not share a Linkage.** Empirically it held — KAME has run five HIGHEST sites
(the NMR pulser, the realtime DSOs, NI-DAQ, DigilentWF) without the collapse the
`-P` sweep shows, because those threads commit disjoint subtrees.

It does not hold. `XSecondaryDriverInterface::onConnectedRecorded`
(`kame/driver/secondarydriverinterface.h`) breaks it by construction:

* it is connected to `onRecord` **with no flags** (`:215`), so it is an immediate
  listener;
* `XDriver::record()` marks the talker (`driver.cpp:52`
  `tr.mark(tr[*this].onRecord(), this)`), so the dispatch happens when
  `finishWritingRaw`'s transaction commits — **inline, on the primary driver's
  acquisition thread**;
* that thread is at HIGHEST (`AcquisitionPriority`, plus the five pre-existing
  sites);
* and the function's first act is `Snapshot shot_all_drivers(*m_drivers.lock())`
  — **the entire driver list** — re-taken on every iteration of its `for(;;)`
  retry loop via `newTransactionUsingSnapshotFor`.

So two acquisition threads each running a secondary driver's analysis — an NMR
pulse analyzer on a DSO, an ODMR analysis on a camera, i.e. exactly the two
drivers wired for `AcquisitionPriority` — contend at whole-driver-list scope at
HIGHEST. That is the regime measured at 10x throughput loss for four such threads
and 42x for eight.

### Fix: the fan-out point lowers itself

`onConnectedRecorded` now opens with
`Transactional::ScopedPriority(Priority::NORMAL)`. The commit dispatch cannot be
separated from the commit (`tr.mark` sends on `commit()`), so the priority has to
drop on the *other* side of the boundary — and that is the right side anyway:
**secondary-driver analysis is not realtime work and should not inherit HIGHEST
merely because a realtime thread invoked it.**

The general rule this instantiates: **a listener that widens the scope it touches
should drop the priority it was entered at.** Worth applying to any future
immediate listener that snapshots an ancestor.

Audited the other `onRecord` listeners for the same shape:

    kame/forms/driverlistconnector.cpp:101      FLAG_MAIN_THREAD_CALL — deferred, safe
    modules/nmr/.../pulserdriverconnector.cpp:31 FLAG_MAIN_THREAD_CALL — safe
    kame/analyzer/recorder.cpp:60               immediate, but uses the passed
                                                shot; no ancestor snapshot
    kame/analyzer/analyzer.cpp:398              immediate; snapshots itself and
                                                the source entry, both leaf-ish,
                                                not the driver list

Only the secondary-driver interface fans out to the list, so it is the only site
that needed this.

### The general fix: realtime ends with the record

Patching the secondary-driver interface was treating a symptom. The rule (user) is
that a primary driver must be back at NORMAL after `record()` — everything
downstream is somebody else's work. Two places implement that, because the
downstream work is split across the commit boundary:

* **`Transaction::finalizeCommitment`'s messaging loop** (kamestm). `XDriver::record()`
  *marks* the talker, so `onRecord` listeners are dispatched inside the commit and
  cannot be reached from outside it. The loop is now wrapped in
  `ScopedDemoteRealtime`. Two lines above it, this function already does
  `m_oneup.release(); // yield the running slot before messaging` — shedding a
  realtime priority is the same idea, for a sharper reason. This one guard covers
  every marked-message listener at once: the secondary-driver chain, the
  scalar/calibrated entries, the recorders, `onVisualization`.
* **`XPrimaryDriver::finishWritingRaw`** after the commit, for `visualize()` and
  the `onVisualization` talk, which are plain calls outside it.

`ScopedDemoteRealtime` is **one-directional by design**: it demotes HIGHEST and
leaves everything else alone. Raising a lowprio committer to NORMAL would be an
escalation path, not a fix — a script or a redraw would dispatch its listeners at
a priority it cannot claim itself.

That is not hypothetical. The first version of the secondary-driver patch used
`ScopedPriority(Priority::NORMAL)`, which raises as readily as it lowers, and
`requestAnalysis()` is reachable from `xpythonmodule.cpp:972` — i.e. from the
Python thread. It would have escalated script-initiated analysis on every call.
The secondary-driver guard is still needed alongside the general one, because
`requestAnalysis()` calls `onConnectedRecorded` directly rather than through a
marked message.

## What HIGHEST actually buys, and the budget that closes the hole it left

Demotion shrinks HIGHEST's reach, and the observation that follows (user) is
correct: with the listeners still on the acquisition thread, HIGHEST buys much
less than it looks like.

What is left is **the contention window** — loop top through CAS success: the
settings Snapshots (`***node()`), `Node::snapshot()` for the ctor and each retry,
and the bundle/commit chain. That is where starvation lives, so nothing the thread
*needed* was given up: the demotion only releases the part after it has already
won.

What was given up is **the period**. The dispatch runs on this thread at NORMAL, so
a slow secondary-driver analysis delays the loop's next iteration, and at NORMAL it
can wait. HIGHEST protects the contention window; it does not protect cadence.
(Not a regression — before `AcquisitionPriority` those threads were NORMAL and the
dispatch was NORMAL too. HIGHEST is a strict improvement on NORMAL, just a smaller
one than it appears.)

### The wait budget closes it, and this corrects an earlier claim

Earlier in this file: *"a budget is inert on HIGHEST"*. That holds only **while it
is HIGHEST**. The moment `ScopedDemoteRealtime` drops the thread to NORMAL, the
budget binds — so a budget on a realtime acquisition thread is not useless, it is
precisely the tool for the demoted region.

And because the budget is an **absolute thread-local limit** rather than a
per-scope duration, one guard at the top of `finishWritingRaw` covers both demoted
regions at once: the marked-message dispatch inside the commit
(`finalizeCommitment`'s messaging loop, which kamestm cannot give a policy value
to) and `visualize()` / `onVisualization` after it.

    XPrimaryDriver::downstreamWaitBudgetUS()     virtual, default 20 ms

So the two mechanisms now do exactly what each is for, and neither substitutes for
the other:

    demote HIGHEST     downstream does not impose on others
    wait budget        downstream does not block the realtime loop's period

This is also the first real user of `ScopedWaitBudget`, which had zero call sites
and was recorded as a feature without a consumer. Its consumer turns out to be the
realtime loop bounding the non-realtime work it must wait for — not, as first
guessed, a driver bounding its own commit.

### Why the default is 20 ms and not 0, and not gated on HIGHEST

Shipped first as `default 0 = unbounded`, with the reasoning that the value comes
from the acquisition cycle and is therefore the deployment's to pick. Then, offered
a 20 ms default, I proposed arming it **only** for a thread that entered at
HIGHEST — since on a NORMAL thread the guard binds the record commit too, not just
the demoted downstream, and the measurement below shows that is not free.

Both were wrong, and the correction is a domain fact, not a tuning preference
(user): **past roughly 20 ms a stalled record starts to distort the measurement,
and that is as true at NORMAL as at HIGHEST.** KAME is an instrument. A record whose
commit sat for a third of a second is a bad data point, not a slow one. So the bound
is not a realtime feature to be gated on priority — it is the acquisition path's
contract, unconditional, 20 ms.

Grand-scope arm, 8 threads:

                   throughput   p99.99    p99.999   MAX
        no budget    2.36 M/s   3.67 ms   67.1 ms   326.6 ms
        20 ms        2.25 M/s   16.8 ms   21.0 ms    20.3 ms

−4.7 % of commit throughput: a clipped commit stops waiting and retries, and the
retry adds CAS pressure. 8-of-8 and 1-of-8 budgeted measured 2.25 vs 2.26 M/s, so
that cost is the clipping itself and not a cascade through the other threads.

**And my reading of the p99.99 was wrong too.** I reported "4.6× worse p99.99
(3.67 → 16.8 ms)" as a cost. Against the criterion that matters — nothing over
20 ms — 16.8 ms is *inside* the budget. The budget does not thicken the tail past
its own line; it compresses everything above it down onto it. Read correctly, the
budgeted row has **every percentile including MAX under 20 ms**, which is the whole
property being bought. Throughput is the only thing actually traded, and for a
measurement path that is the right direction to trade.

Generalisable: a percentile moving *within* a declared bound is not a regression
against that bound, and quoting it as one argues against the very guarantee being
added. Compare against the requirement, not against the unbounded baseline.

No record is lost either way — the budget bounds *waiting*, and the clipped commit
retries through `iterate_commit` until it succeeds. The failure mode is CPU spent
retrying instead of sleeping, which is why the number wants to stay comfortably
under the acquisition period: then a blown budget costs a late record rather than a
lost one. A driver with a period near or under 20 ms should override it downward;
`return 0` disables.

## The `kame/` side now at least compiles — and how, since there is no build here

Every `kame/` and `modules/` change in this file shipped **uncompiled**: this
session has no Qt Creator build, and `kamestm/tests/` is a Qt-free harness, so
`ctest` passing said nothing about the host side. The starvation handler in
`main.cpp`, `AcquisitionPriority`, the in-transaction interface detector, the
`queryStatus` refactor, the pybind changes, the `ScopedDemoteRealtime` sites and
`downstreamWaitBudgetUS()` were all reasoned-about, not built. One of them
(`modules/python/basicdrivers.cpp`, missed by a directory-scoped grep during the
`queryStatus` refactor) had already broken the build once.

A full build is not needed to close most of that — `clang++ -fsyntax-only` is,
once three build-system inputs are supplied:

* `-DVERSION=... -DKAME_MODULE_DIR_SURFIX=... -DPACKAGE=...` — qmake passes these;
  without them `main.cpp` fails with *undeclared identifier* and then a cascade.
* **uic output.** `#include "ui_*.h"` is generated, so run it first:
  `for ui in $(find kame modules -name '*.ui'); do
  $QTDIR/libexec/uic "$ui" -o gen/ui_$(basename ${ui%.ui}).h; done` (54 headers).
* **Qt as frameworks on macOS**: `-iframework $QTDIR/lib` plus one
  `-I $QTDIR/lib/Qt<Module>.framework/Headers` per module. Plain `-I $QTDIR/include`
  does not resolve `<QString>`.

And one trap worth recording: do **not** pass `-D slots= -D 'signals=public'`
here, even though CLAUDE.md gives them for checking a header in isolation.
CPython's `object.h` has a real member named `slots`, so with pybind11 in the
translation unit those defines produce *expected member name* errors in
`Python.h` — the mirror image of the hazard they exist to catch. The defines are
for Qt-free headers; a TU that includes `<QObject>` for real does not need them.

Result — all 15 touched translation units pass with no errors (two pre-existing
`-Winconsistent-missing-override` warnings from `DEFINE_TYPE_HOLDER`, unrelated):

    kame/       main, primarydriver, secondarydriver, interface, xpythonmodule,
                xpythonsupport, x2dimage, analyzer
    modules/    optics/core/digitalcamera, dso/core/dso, dcsource/core/dcsource,
                dcsource/userdcsource, relay/core/relaydriver,
                python/basicdrivers, tempcontrol/tempcontrol

`-fsyntax-only` is not a link, so it does not prove the `queryStatus` overrides
match their bases across every module, nor that anything *runs*. It does close
the class of error that has actually bitten here — a missed call site, a
mistyped member, a wrong signature — for every file this work touched.

## One detector, two call sites: `isInTransaction()` / `gWarnIfInTransaction()`

`XInterface::lock()` got a debug-only "you are inside a transaction" report
earlier in this work. Adding `msecsleep()` to the list, I wrote the machinery a
second time — its own gate, its own deduplicating set, its own abort environment
variable, its own message assembly. Correctly called out (user) as inelegant: the
remedy is to publish the *predicate* and share the *reporter*, not to copy them.

    Transactional::isInTransaction()               the predicate, published once
    Transactional::warnIfInTransaction(what, ...)  the one report body (debug-only)
    gWarnIfInTransaction(what)                     kame/support.h wrapper, fills in
                                                   __FILE__:__LINE__

On the naming, which was asked about: `isDuringTX` is not idiomatic English —
"during" wants an event, not a state, and `TX` reads as an abbreviation nobody
outside this file would expand. `isInTransaction()` is the ordinary phrasing.
The kame-side wrapper follows `gErrPrint`/`gWarnPrint`, and is a macro for the
same reason `gErrPrint` is: only a macro can capture the caller's source line.

Deduplication key: the wrapper passes `__FILE__ ":" __LINE__`, which is both the
key and a printable location. `msecsleep` has no source location to offer, so it
passes its caller's return address instead, printed as a pointer for `atos` /
`addr2line`.

### Why `msecsleep` still goes through a function pointer

It cannot call `isInTransaction()` directly. `xtime` must know nothing about
transactions, and more concretely: `detail::s_tx_nest` is *defined* in
`transaction_impl.h`, which `mutex_test`, `atomic_queue_test` and the
pool-allocator tests never include. A direct call would make those binaries fail
to link. So `xtime.h` exposes `g_sleep_in_transaction_reporter` plus a
`ScopedSleepInTransactionOK` suppression, `transaction_impl.h` installs a
three-line adapter into it at static-init time, and the adapter calls the shared
reporter. The pointer stays null in binaries without the STM.

kamestm has two legitimate in-transaction sleeps, both suppressed at the call
site: the out-of-memory backoff in `print_recoverable_error` (it *is* the delay,
and it is called from inside the transaction it delays) and lazy TSC calibration
in `timeStampCountsPerMilliSec` (one-time, and the first `timeStamp()` can fall
inside a transaction).

### The static and the dynamic check cover different things

`tools/audit/check_stm_closures.py` already flags a literal `msecsleep(` inside an
`iterate_commit` closure — that was there before, in `SIDE_EFFECT_RE`. The runtime
detector exists for the two cases a source scan cannot reach: a sleep several call
levels *below* the closure, and a sleep anywhere else in a transaction's lifetime.
Neither subsumes the other; the static one needs no debug build and no execution,
the dynamic one needs no call-graph.

### Two ways the verification of this nearly fooled me

* **A debug-only check in a Release test tree tests nothing.** The first build of
  the new `transaction_sleep_in_tx_test` "passed" — `CMAKE_BUILD_TYPE=Release`
  means `NDEBUG`, so the detector and the whole test compiled to nothing. The
  target now carries `-UNDEBUG`, and the `#ifdef NDEBUG` arm of the test *fails*
  rather than skipping, so the flag cannot be silently lost.
* **`__builtin_return_address(0)` is only the caller's address while the frame is
  real.** The standalone harness had `msecsleep` `inline` in
  `support_standalone.h`; at `-O3` it inlined, and every call site reported the
  same libsystem address — two distinct sites counted as one. Fixed by making the
  harness's `msecsleep` out-of-line, matching the shape of the real `xtime.cpp`.
  Then the deduplication case *still* failed at +2, because `-O3` **unrolled** the
  two-iteration loop into two distinct return addresses. There the detector was
  right and the test was wrong: the case now calls one `noinline` function twice.

The test also pins that a **Snapshot** alone does not trip the detector. That is
the intended semantics — a Snapshot blocks nothing — and it is exactly what
`s_tx_nest` gives, being held for a Transaction's whole lifetime but only during a
Snapshot's construction.

## OS priority is policy, not mechanism: `setOSPriorityHook`

Asked (user), with PREEMPT_RT support on the horizon: *shouldn't the current
STM-priority → OS-priority coupling change?* Yes — and the Windows measurement
above already showed why in miniature. `setCurrentPriorityMode` contained a
Windows-only arm (pre-existing, `59d942f36`) mapping HIGHEST ↔
`THREAD_PRIORITY_TIME_CRITICAL` inside kamestm itself. Three things are wrong
with that once an RT Linux port is real:

* **The mapping is a deployment decision a library cannot make.** On PREEMPT_RT
  the numeric level is chosen relative to the kernel's threaded irqs (default 50)
  and ksoftirqd; the policy might be `SCHED_FIFO`, `SCHED_RR` or
  `SCHED_DEADLINE` (which has no static priority at all); and raising it needs
  `CAP_SYS_NICE` or an `RLIMIT_RTPRIO` grant, so the call can *fail* and policy
  decides what that means. Hardcoding any of it into the STM would be exactly the
  "RT-only design mixed into the general code" this work is required to avoid.
* **Documented RT practice is set-once, not toggle-per-record.** POSIX RT
  scheduling attributes are set at thread setup (`pthread_attr_setschedparam`,
  explicit-sched); Windows' own low-latency path (MMCSS) likewise registers a
  thread once. A hidden per-record `pthread_setschedparam` issued from inside an
  STM commit would be a surprise to anyone auditing an RT deployment.
* **A standalone library silently promoting host threads was already a smell.**
  Any Windows program linking kamestm and using `Priority::HIGHEST` got
  TIME_CRITICAL whether it wanted it or not.

The change mirrors `setStarvationHandler` exactly — the host installs policy,
the library provides the call site:

    Transactional::setOSPriorityHook(hook)   null by default; called by
                                             setCurrentPriorityMode with the new
                                             priority, on the changing thread

The hook type is a `noexcept` function pointer because it is reached from
`ScopedDemoteRealtime`'s noexcept destructor. With it, the STM core's only
`<windows.h>` dependency is gone.

**Windows behaviour is preserved where it belongs**: `kame/main.cpp` installs the
historic mapping as the hook, with a `thread_local` skip — every priority except
HIGHEST maps to `THREAD_PRIORITY_NORMAL`, so transitions among NORMAL / SCRIPTING
/ UI_DEFERRABLE / LOWEST no longer pay a no-op syscall (previously *every*
`setCurrentPriorityMode` call on Windows was one). One deliberate subtlety: the
skip means an OS priority set externally on a thread is left alone until HIGHEST
is involved, where the old arm forced NORMAL on every call.

**The PREEMPT_RT plan this enables** (a plan, not an implementation — no RT host
here): leave the hook null. The acquisition thread's OS class is set once at
thread start by the deployment; `ScopedDemoteRealtime` then moves only the STM
priority, and whether downstream listeners may run at FIFO for their (bounded by
the wait budget, at NORMAL STM priority) duration — or whether a hook should
toggle the OS class too — is the deployment's call, made in one visible place.

Also gated in the same commit: `finalizeCommitment`'s demote guard now skips when
`m_messages` is empty, so a listener-less commit — most settings commits — pays
neither the guard nor, with a hook installed, its two syscalls.

Non-RT regression check: with a null hook `setCurrentPriorityMode` is the same
TLS store as before on macOS/Linux (12/12 ctest, audits clean); on Windows the
KAME application installs the old mapping before any driver thread exists. The
hook install and the mapping itself sit in an `#if _WIN32` arm this host cannot
compile — same standing caveat as every Windows-side line in this work.

**Superseded the same day, before ever being pushed — see the next section: the
hook is gone again.** The layering argument above stands; what was wrong is the
behaviour any installed hook would produce.

## Correction: OS priority is a thread property, not a transaction property

The hook was the right *layering* and the wrong *behaviour* (user): an OS
scheduling class should be **permanent for the thread**, not toggled with STM
priority changes — and once it is permanent, there is nothing for kamestm to
call, so the implementation belongs to KAME. The hook lasted one commit and was
removed unused rather than left as an attractive nuisance (the A/B/C/D lesson:
an API whose only use case has been judged wrong will rot).

The argument is not just "set-once is the documented practice" (it is — POSIX
RT attributes at thread setup, MMCSS one-time registration). It is an RT
argument: with the OS class coupled to `ScopedDemoteRealtime`, every
acquisition cycle handed the CPU to arbitrary threads for its entire demoted
downstream half — listeners, `visualize()` — right when the loop is racing the
next trigger. Being preempted there eats period margin unpredictably, which is
backwards: the loop should finish its whole iteration at acquisition priority
and yield *naturally* in the device wait, where it blocks and the CPU frees
anyway. The demotion's real job was never CPU allocation:

    ScopedDemoteRealtime   STM-level.  Stays.  Prevents an immediate listener
                           that widens scope from negotiating at HIGHEST and
                           putting two realtime threads on one Linkage.
    OS scheduling class    thread-level, thread-lifetime.  KAME-side.

So now:

* `Transactional::setCurrentPriorityMode` is a pure TLS store on every
  platform. kamestm has **zero** OS-scheduler awareness — no windows.h, no
  hook. (The brief hook, `eab100ec8`, never reached the remote.)
* `AcquisitionPriority` (kame/driver/primarydriverwiththread.h) raises the OS
  class in its constructor and restores it in its destructor — the RAII spans
  the acquisition loop, which spans the thread, so this *is* set-once. The OS
  half lives in `raiseAcquisitionOSPriority_()` / `restoreAcquisitionOSPriority_()`
  (primarydriver.h/.cpp): Windows `THREAD_PRIORITY_TIME_CRITICAL`, no-op
  elsewhere, and the single visible place where PREEMPT_RT's
  SCHED_FIFO/RR/DEADLINE decision goes when it comes.

**Deliberate Windows behaviour change** (the historic arm toggled): the demoted
downstream now runs at TIME_CRITICAL. That is the point — the STM priority
drops, the CPU stays. A listener too long to tolerate at acquisition priority
was already too long for the acquisition loop, and the wait budget, the
`FLAG_MAIN_THREAD_CALL` rule and the record-commit telemetry are the tools for
noticing it.

Restore goes to `THREAD_PRIORITY_NORMAL` rather than a saved value, on the
grounds that acquisition threads are created for the loop and die with it. The
same reasoning says nesting `AcquisitionPriority` twice on one thread would
restore early — it has no reason to ever nest.

## Should HIGHEST use privilege among its own tier? Measured: no — tags suffice

Asked (user), given that KAME now really deploys HIGHEST: *should tag/privilege
work between HIGHEST threads, with NORMAL subordinated to HIGHEST's
tag/privilege — or, since HIGHEST is not supposed to starve, are tags alone
enough?*

**What the code already does.** Tags are unconditional on the retry path
(`transaction_impl.h`: "the retry-path tag_as_contender call sites are now
unconditional") — a HIGHEST contender is counted in `sig_C`, participates in the
age-ordered stamp preemption (older-always-wins), and is seen by the owner-skip
lease. The privilege *claim* is on a path HIGHEST can reach in principle, but
gated behind the livelock probe (`_ll_saw`), which a promptly-winning spinner
never trips — measured 0.000 claims. And `fair_mode_blocks_me` is consulted
below the round-loop-top HIGHEST breakout, so HIGHEST ignores everyone's
privilege. So "tags only" is not a proposal; it is the present design.

**The measurement** — the *forbidden* deployment (two+ HIGHEST on one linkage),
worst-case grand scope at 100 % duty, M3, 4 s runs, per-thread split added to
the latency bench for this question:

    -t 2 -P 2 (two spinners, nothing else; 3 reps)
        thr#0 / thr#1 balanced within 1 % (e.g. 3.48 vs 3.45 Mcommit/s)
        p99.99 = 57–98 µs, STM-attributable MAX sub-ms
        (one rep showed 60–68 ms MAX on BOTH threads at once: OS preemption,
        not STM starvation — correlated across threads.)
    -t 8 -P 2 (plus six NORMAL)
        HIGHEST thr#0/#1 balanced within 5 % (0.41 / 0.39 Mcommit/s),
        p99.99 = 163–327 µs
        NORMAL group: ~8 k commits per 4 s vs HIGHEST's 3.2 M — mean ~3 ms,
        MAX 81–183 ms
    aggregate cost of the violation: 2.40 -> 0.80 Mcommit/s (3x)

**Verdict: tags suffice; privilege for HIGHEST would make it worse.**

* Privilege is a shield for a thread that *yields* — it protects a sleeper from
  being starved while it waits its turn. HIGHEST never yields, so it has
  nothing to shield. Between exactly-two spinners, CAS linearization already
  hands one of them the win each collision round; the measured alternation is
  the theory working.
* Granting HIGHEST privilege would convert "loser retries and usually lands in
  the winner's gap" into **strict serialization behind the holder — including
  any OS preemption of the holder**. Today a 60 ms preemption of one spinner is
  60 ms of free run for the other; under privilege it would be 60 ms of spinning
  behind a stamp. On a normal OS that worsens the RT tier's tail, and it adds a
  waiting relation inside the RT tier that the TLA+ liveness model does not
  have — re-verification surface spent on the case the deployment contract
  forbids anyway.
* Structural NORMAL subordination (wait on a HIGHEST stamp) has the same trap:
  NORMAL's ~8 k commits above are exactly the gap-sneaking that a stamp wait
  would forbid. NORMAL's *bound* never depended on beating HIGHEST — it is the
  wait budget; its *completion* depends on HIGHEST's duty cycle either way, and
  100 % duty is synthetic (real acquisition loops block in device waits).

**One consequence for the doctrine.** The "no two HIGHEST on one linkage"
deployment invariant is hereby *downgraded*: it is not a liveness precondition
(no starvation, no livelock — measured), it is a **throughput contract** (3x).
HIGHEST between spinners is lock-free, not wait-free; its per-thread bound is
statistical (geometric tail, p99.99 in the 10^2 µs range) — which is also its
exact status against NORMAL churn even *with* the invariant, since a NORMAL
commit invalidates a HIGHEST snapshot all the same. The invariant buys
throughput and tightens the tail; it does not buy the liveness it was earlier
assumed to carry. A debug-time detector for two HIGHEST negotiating one linkage
is accordingly a *performance*-bug detector, and still worth having.

### Does a privileged NORMAL yield to a HIGHEST tag? — the interaction matrix

Asked (user) as the natural follow-up to the verdict above. The letter-answer is
**no — and it could not**: the stamp carries exactly one priority bit
(`STAMP_LOWPRIO_MASK`, set for the three revocable levels, sealed entirely under
`KAME_STM_COMPACT_STATE`), so a HIGHEST tag is bit-identical to a NORMAL tag.
Nothing on the linkage can key on "the contender is HIGHEST". But the intent
behind the question — *can NORMAL privilege delay an acquisition thread?* — is
answered by construction, in four layers:

1. **Privilege never blocks HIGHEST.** `fair_mode_blocks_me` is consulted below
   the round-loop-top HIGHEST breakout. A privileged NORMAL vs a HIGHEST is two
   non-sleeping CAS racers — the same benign alternation measured in the
   two-spinner run (p99.99 = 57–98 µs). The privilege does not need to be
   yielded because it was never in HIGHEST's way.
2. **A NORMAL's privilege actually helps the HIGHEST.** It blocks the *other*
   NORMAL/lowprio contenders via fair-mode, reducing the HIGHEST's opposition to
   one thread and lowering `sig_C` churn.
3. **Age arbitrates the stamp slot, not priority** — the symmetric preempt
   window (user-designed, `KAME_STM_PREEMPT_WINDOW_US` = 100 µs): an *older*
   HIGHEST's tag respects a younger privilege holder's burst window, then
   preempts the Reserved stamp; the holder's preempt-recovery clears
   `m_registered_privileged` — privilege revoked by age, the TLA+ older-wins
   axis. A *younger* HIGHEST leaves the slot alone and just keeps racing.
4. **The structural subordination exists — dormant, probe-gated.** HIGHEST is
   not excluded from the livelock probe or the claim path (its age floor is
   `KAME_STM_PRIV_AGE_NORMAL_US`, same as NORMAL). A HIGHEST that genuinely
   stalled would claim Reserved, and `fair_mode_blocks_me` is priority-blind
   (TID compare; non-lowprio stamps never expire), so every NORMAL would then
   yield to it structurally. Measured grants = 0.000 means this ladder has
   never been needed, not that it is missing.

Making the privileged NORMAL *actively* step aside on a HIGHEST tag would
require adding a HIGHEST bit to the stamp (five consumers plus the
COMPACT_STATE seal to re-verify) in order to void the privilege exactly when
the probe had just certified its holder as stalling — re-creating the
starvation privilege exists to cure, to speed up a race the HIGHEST is not
delayed by in the first place.

### "If LOWPRIO is unused, make it the HIGHEST bit"? — it is not spare

Proposed (user) against the section above. The premise does not hold: in the
deployed (64-bit) build the lowprio bit is load-bearing for two shipped
mechanisms —

* **The 51 ms privilege revocation.** `stamp_is_expired_lowprio` keys on it,
  with three consumers that must agree (`fair_mode_blocks_me`,
  `try_register_privileged_tidstamp`, `i_am_privileged_now`). Remove it and a
  stuck SCRIPTING / UI_DEFERRABLE / LOWEST holder leaves what the code's own
  comment calls "a frozen Linkage nobody can overwrite" — the failure whose
  terminal form is the HANG-watchdog abort, and the property that made those
  priorities "revocable" in the first place.
* **The starvation bound's gate.** `throw_if_starved_` reads
  `stamp_is_lowprio(shot.m_started_time)` off the transaction's own stamp —
  chosen deliberately over TLS so the check costs nothing on the fast path and
  reflects the priority the operation *started* at. The revocation and the
  timeout are two halves of one design: privilege can be taken away, therefore
  there is a way to fail.

The bit is "unused" only under `KAME_STM_COMPACT_STATE` — and that mode exists
for 32-bit no-DCAS hosts where the stamp is `[us:24|tid:8]` in an `int32_t`:
no priority bit of any kind fits there, HIGHEST included.

Nor is there a spare bit to add instead: the 64-bit stamp is exactly full,
`45 (µs) + 1 (lowprio) + 2 (kind) + 16 (tid) = 64`, and all four kind values
are taken — `Reserved` (= 3) *was* the spare, already reclaimed for per-Linkage
privilege. Sharing the lowprio bit ("set = not NORMAL") corrupts both
consumers: HIGHEST privilege would expire at 51 ms and, worse,
`throw_if_starved_` would throw on HIGHEST — the one tier that must never get a
starvation timeout.

If a justified consumer ever materialises, the honest door is stealing one µs
bit (45 → 44 still wraps at ~200 days, modular comparisons safe far below
half-range). Today there is no such consumer: the only proposed use — NORMAL
yielding structurally to HIGHEST — was measured and rejected above, and the
two-HIGHEST *detector* does not need a stamp bit either. `Linkage`'s
`PriorityState` is `{tid, lease_us, start_us}` — despite the accessor's name
(`loadPriority`) it records no `Priority` — so the detector's natural shape is
a debug-only (`#ifndef NDEBUG`) per-Linkage field written by HIGHEST
contenders, costing the release layout nothing.

## Rule 0: HIGHEST strips a stuck foreign privilege — the bundle-protocol hole

The interaction matrix above said a privileged NORMAL was never in HIGHEST's
way. Objected to (user), correctly: that conclusion came from the *leaf-
symmetric* measurement. In the **bundle protocol** the pair is asymmetric — a
wide-scope HIGHEST must re-bundle O(N) on every disturbance while the holder
redoes O(1) — and it is the one pairing with **no yielding mechanism at all**:
HIGHEST never consults fair-mode (round-loop breakout), the holder never sleeps
(that is what privilege means), its privilege ends only with its own commit,
and the tag rules 1–4 key on age, so a *younger* HIGHEST never preempts. In the
no-winner pathology (mutual bundle/unbundle invalidation, the hard-link
CAS-never-succeeds shape) nothing breaks the tie. The remedy (user): HIGHEST
forcibly strips the privilege and tags itself — **which requires knowing the
holder is not HIGHEST**, since stripping a fellow HIGHEST's probe-gated
escalation would invite strip wars inside the RT tier. One turn earlier this
file said the HIGHEST bit had "no justified consumer"; this is the consumer.

### The mechanism is a side word, not a stamp bit (user's design)

The stamp cannot carry it (layout full, lowprio bit load-bearing — previous
section), and stealing a µs bit would touch every stamp consumer. Instead:
`Linkage::m_priv_owner_prio`, `[15:0] = holder tid, bit 16 = claimed at
HIGHEST`. The race analysis that makes it sound, and answers "tid を CAS して
から prio を CAS? race ある?":

* Two separate atomics would race — a reader could pair A's tid with B's
  priority. **One packed word removes the pairing race, and no CAS is needed
  at all**: only the thread whose own plain stamp occupies the slot may
  upgrade it to Reserved, so writers are already serialized by slot ownership.
* The claimant release-stores the word **before** its Reserved CAS. A reader
  that acquire-loads the stamp and sees Reserved(A) therefore sees A's word.
* The reader validates `tid(word) == tid(stamp)`; any mismatch — claim gap,
  epoch change, global-privilege mode (which never writes the word) — reads as
  "unknown: do not strip". Every residual race degrades toward not stripping;
  none can strip a HIGHEST holder.

### Stripping on sight measured NET NEGATIVE — the patience gate

The first implementation stripped on first encounter. Interleaved A/B (grand,
`-t 8 -P 1`, 5 reps): aggregate 2.37 → 2.26 Mcommit/s (−4.6 %), HIGHEST p99.9
1.5 → 2.6 µs, **no** tail win, 183 k strips per 4 s. The reason was already
written in the interaction matrix and I failed to apply it: *a NORMAL's
privilege helps the HIGHEST* — while held, fair-mode silences every other
NORMAL, thinning the HIGHEST's opposition to one thread. Stripping on sight
destroyed exactly that thinning and returned the pack to churn. And the common
case needs no strip at all: privilege is per-transaction, ends at its commit —
a healthy holder holds for microseconds; base HIGHEST p99.99 was already
7–12 µs.

So Rule 0 is **patience-gated**: a HIGHEST strips only a holder it has been
stuck behind — same Reserved episode, tracked per-transaction — for
`KAME_STM_PREEMPT_WINDOW_US` (100 µs, the constant the burst window already
uses). Re-measured: parity with base on every metric (p99.9, p99.99, MAX,
aggregate, `-P 0`, `-t 2 -P 2`), and **zero strips in every benchmark run** —
the healthy holder is never touched, and 100 µs bounds HIGHEST's exposure to
the pathological one.

### Proving both halves

Zero strips proves the zero-cost half only. The insurance half cannot be
manufactured through the public API (claims are probe-gated), so
`transaction_priv_strip_test` — built with `-fno-access-control`, deliberately
white-box — plants a synthetic foreign Reserved stamp plus side word on the
Linkage and pins all four arms: stuck non-HIGHEST holder → stripped after the
window; holder marked HIGHEST → untouched; side-word tid mismatch → untouched
(unknown is conservative); patience not elapsed → untouched. 13/13 ctest.

Also fixed while here: `g_priv_strips` (always-on relaxed counter) so a plain
build can verify the mechanism fired; the latency bench prints it with `-P/-L`.

### NORMAL-only workloads: unchanged in principle, and what "in principle" means

Asked (user). The *decision logic* is structurally unreachable without a
HIGHEST thread: Rule 0 is gated on `getCurrentPriorityMode() == HIGHEST`, so in
a NORMAL-only process no strip, no counter bump and no side-word read can
occur, and rules 1–4 / fair-mode / claim / expiry execute exactly the old
instructions. What is *not* zero is the executed-instruction delta, and each
item is incapable of changing a branch outcome:

* two zero-initialisations (+16 B) per Snapshot construction — the patience
  memory, read only inside the HIGHEST-gated block;
* one release-store per tagged linkage at privilege claim — **NORMAL claimants
  write the side word too**, deliberately: the word must already be correct at
  the instant a HIGHEST first appears, which is what makes the tid validation
  sound (a write-when-HIGHEST-appears scheme would race against exactly the
  reader it serves);
* one TLS read + store when a privileged transaction extends Reserved to a new
  linkage, and one TLS read when any tagger meets a Reserved stamp (the
  short-circuited right operand of `_cur_is_priv && ...`);
* +4 B (8 B with padding) per Linkage.

Under `KAME_STM_COMPACT_STATE`, `is_priv_stamp` is constant-false and Rule 0 is
dead-code-eliminated entirely. Empirical cross-check of precisely this
question: the `-P 0` interleaved A/B (parity) and 13/13 ctest.

### Rule 0 and `ScopedDemoteRealtime`: the demotion's justification, corrected

Asked (user): with Rule 0, does KAME's HIGHEST still need the demotion to
NORMAL after the record? My first answer defended it with the 3× aggregate and
the 8 → 163–327 µs tail from the `-P 2` runs. **Rejected (user), correctly: the
3× has no basis here.** Those are 100 %-duty synthetic-spin numbers; a real
deployment's collision probability scales with duty (µs commits × kHz rates ≈
10⁻²–10⁻³) and each collision costs one peer-TX length. The numbers do not
transfer, and quoting them as the demotion's justification was wrong.

The correct principle (user): **if it is clear a HIGHEST TX contains no
msecsleep, no lock and the like, it cannot starve anything and cannot be
starved.** Optimistic STM holds nothing during a transaction — a clean TX is
visible to others only as a CAS loss at its commit instant, so the loser's
delay is bounded by the peer's TX length, at any priority. The two-spinner
measurement (balanced alternation) was this principle observed, not a
surprising discovery.

What the demotion's justification then reduces to: **making the antecedent
constructively true for code the driver author cannot vouch for.** Split by
tier:

* C++ listeners: the antecedent is machine-checkable — rule-5 static audit,
  `gWarnIfInTransaction` on interface locks, the msecsleep detector, the
  foreign-lock guard. "明確" is achievable.
* Python-involved paths: the GIL is a lock structurally inside the TX, so the
  antecedent cannot hold — **but the demotion does not guard that boundary
  anyway**: math-tool functors run inside `analyzeRaw`, upstream of the
  demote, at HIGHEST today. And a GIL-holding TX at HIGHEST never sleeps in
  negotiation, so the rule-4 deadlock shape (GIL holder blocking in
  negotiation) becomes less reachable, not more.

Rule 0's role is unchanged by this correction: it caps the demoted-NORMAL
(or any privileged-NORMAL) holder at 100 µs in the no-winner pathology, and
is indifferent to whether kame demotes.

So the demotion is **not load-bearing for starvation-freedom**; it is a
policy choice about whether unvouched code runs in the RT tier. Whether to
keep it, drop it, or turn it into a per-driver vouch is the deployment's
call, not a correctness requirement.

**Decision (user): status quo — the demotion stays.** The deciding fact is
Python: the downstream can reach it (secondary drivers invoking a Python
driver's analysis, `onVisualization` callbacks, math-tool functors), and the
GIL is a lock structurally inside those transactions, so the clean-TX
antecedent cannot be made true for the downstream *as a class* — no audit or
detector can vouch it. Code that can reach the GIL does not run in the RT
tier; that is now the demotion's one justification on record, replacing both
withdrawn ones. No per-driver vouch virtual either — same reason, a driver
author cannot vouch what their listeners' listeners do.

Known residual, accepted as-is: `analyzeRaw`'s math-tool functors take the GIL
at HIGHEST *upstream* of the demote, inside the record commit. That is the
driver author's own vouched zone — the caller-side-time-management contract —
and unchanged by this decision.

## Field-livelock triage: why "no starvation timeout" is itself a clue

Field report: rare livelock when operating the UI during an NMR measurement,
HIGHEST-ification suspected. Asked (user): why did the UI's starvation timeout
not fire? Verified in code first: **both sides of the bound are armed** — the
plain-Snapshot constructor stamps `m_started_time` with the lowprio bit
(transaction.h:1554) and `Node::snapshot()`'s retry loop bumps
`m_tx_retry_count` and calls `throw_if_starved_` per retry, alongside the
Transaction-side check in `operator++`; and the `XInterface::start()/stop()`
plain `setCurrentPriorityMode(NORMAL)` calls run on their own freshly spawned
XThread, so the main thread's UI_DEFERRABLE (and with it the lowprio stamp
bit) is not clobbered. So on current code a UI transaction or snapshot loop
that starves ≥1 s at ≥8 retries throws XKameError into KAME's existing catch
boundaries.

A hang with *no* timeout therefore means one of exactly three things:

1. the running binary predates the bound or the main.cpp handler (commits are
   from the same arc but not the same push);
2. the stuck point is not an STM retry loop at all — the mutex/GIL class
   (graph OSO mutexes, `kame_mainthread` handshake against a stuck Python
   thread, interface mutex from a rule-6 listener). The bound sees only STM
   retries, and the HANG watchdog needs a single negotiate call to sleep 5 s,
   which spin-retry loops never do. **A silent hang points at non-STM
   blocking**;
3. it fired and a boundary swallowed it into a retry loop — then the message
   log shows the XKameError once per second.

Triage recipe for the next occurrence: `sample kame 5 -file /tmp/hang.txt` —
`_negotiate_internal`/CV frames = STM negotiation, hot `iterate_commit`/
`bundle` frames = CAS livelock, `psynch_mutexwait` = mutex deadlock,
`PyEval_*` = GIL; plus check the message log for the starvation XKameError
and record the build's commit.

The hunt tool (`transaction_priority_mixed_test`) has so far NOT reproduced
any stall: 120 s flat-out with both lowprio threads, and 300 s with every UI
action a root-scope Tx plus four NORMAL drivers, on the field-equivalent
build (pushed tip, no Rule 0) — all PASSED on this M-series host. The
remaining modelled-vs-field gaps: tree size (a real root bundle is ms-scale,
the test's 16-node one is µs — `KAME_MIX_LEAVES` added for this), and
everything the standalone harness cannot host (Qt event loop, GIL, interface
mutexes) — which is exactly the class that a missing timeout points at.

## The T1Mode field abort: the user's diagnosis was right twice

Crash report analysed (SIGABRT, thread 23, `_negotiate_internal` → `abort()` =
the HANG watchdog; every other STM thread asleep in `negotiate_sleep`; main
thread mid-`XNodeBrowser::process()` building connectors). My first two
readings — "seconds-long FFT inside the Tx" (refuted by the user: the stack
only proves where the thread was at the snapshot instant), then "retry storm ×
never-expiring NORMAL privilege" — each contributed a hardening but missed the
trigger. The user's questions found it: *did the UI timeout cause this?* and
*why would RAII not run the destructor?*

**The ghost-stamp leak.** `throw_if_starved_` sits inside `Node::snapshot()`'s
retry loop, which runs during `Snapshot`/`Transaction` **construction**. A
throw there means the object never began its lifetime: unwinding destroys the
fully-constructed members (`~vector` frees the list of linkage pointers), but
the *stamps those linkages carry* are external side effects whose release
exists only in `~Transaction()`'s body and at the constructor's tail — both
unreachable. The orphaned stamp ages forever, is always the oldest contender,
is never preempted (older-wins) and never cleared (only its owner clears it;
`tags_successful_cas` writes the lease word, not the stamp slot) — and the
negotiation protocol lets contenders CV-sleep waiting for an older peer to
finish. Everyone on that linkage waits for a ghost until the watchdog kills
the process. This explains 以前は起こらなかった (the starvation check is new),
the T1Mode reproducibility (it reliably drives the UI past the 1 s bound), and
HIGHEST's irrelevance. Fixed by catch → `drop_tags_n_privilege()` → rethrow in
the two constructors; `operator++` throws were always safe (complete object,
destructor runs).

**The engine, and why connectors must not throw at all.** The timeout's
throw-and-restart cycle is *forever young* under older-wins arbitration — each
restart discards the seniority that would have won — so a contended UI
operation that used to be slow-but-completing became never-completing at
maximal churn. Worse, `XQConnector`'s constructor pushes `shared_ptr(this)`
onto `s_conCreating` before its STM work: a throw shifts the holder pairing
(the next `XQConnectorHolder_` pops the dead entry — use-after-free) and then
escapes into Qt's event dispatch, which does not support exceptions. So for
connector construction the throw is not merely unhelpful, it is a crash of its
own. kame now (a) suppresses the starvation throw for the duration of
`xqcon_create` (`XQConnector_StarvationExempt`, consulted by main.cpp's
handler — construction retries with accumulated seniority, the pre-timeout
behaviour), and (b) gives `XNodeBrowser::process()` a catch-and-back-off (10
ticks) so any remaining XKameError from its snapshots reports once instead of
retrying at timer cadence or reaching Qt.

The privilege-expiry change earlier in this arc stays as defence in depth:
`stamp_is_expired_priv` bounds ANY Reserved holder (NORMAL included, ~51 ms;
side-word-confirmed HIGHEST exempt) so no future not-winning holder can pin
peers into the watchdog. `transaction_priv_expiry_test` pins the predicate
matrix on both agreeing consumers (it FAILED before the fix — aged NORMAL
blocked forever); `transaction_priv_pin_test` keeps the field shape
(budget-expired spinner + fresh-commit burst + third-party NORMAL) as a
behavioural regression net.

## Corrections and decisions after the T1Mode fix landed

**The NORMAL-privilege expiry is reverted (user ruling: 「privilege expiryは
NORMAL/HIGHESTに適用してはダメだ」).** My `stamp_is_expired_priv` — shipped as
"defence in depth" — changed the meaning of the tier table: NORMAL's
never-expiring privilege *is* the completion guarantee. The revocable tiers
have the starvation timeout as their exit; NORMAL has no exit **by design**,
so its shield must outlast any wall clock, and the TLA+ liveness argument
assumes privilege persists until its holder finishes. The field abort's
blocker was an OWNERLESS stamp — a leak, not a live holder — and leaks are
fixed at the source (ctor exception safety), not by taxing live holders.
"NORMAL priv never expires was falsified" in e5b27bf4e's message was wrong:
what was falsified was only the assumption that a Reserved stamp always has an
owner. `transaction_priv_expiry_test` now pins the restored tier rule (aged
NORMAL **still shields**) and would catch the rejected design as a regression;
`transaction_priv_pin_test`'s stall bound moved 5 s → 12 s, since a live
NORMAL holder may legitimately shield for multi-second stretches — only the
ghost-class (watchdog-class) pin is a failure.

**The starvation bound is 10 s now (user: 「１０秒程度にして、ユーザーがデータ
保存する機会を与える」).** The throw lands in constructors and Qt-adjacent
paths that cannot all be made exception-safe, so firing must be rare; the
bound's role shifts from responsiveness to a **last exit before the 3 × 5 s
HANG watchdog aborts the process** — the UI thread unfreezes with an error
telling the user to save, instead of the app dying with the data. Transient
1–2 s stalls now resolve by seniority (older-wins) rather than by a throw
that restarts the transaction forever-young.

**The timeout-retry-loop audit (user: 「タイムアウトでリトライループに陥る
ところがないかのチェックが必要だ」).** Where a thrown XKameError lands, and
whether anything auto-retries:

| path | state |
|---|---|
| main-thread listeners (`SignalBuffer::synchronize__`) | already caught, event consumed — no retry loop |
| connector value slots (`xnodeconnector.cpp`, 7 sites) | already caught per-slot, red text, human-paced retry only |
| connector construction (`xqcon_create`) | exempted from the throw entirely (UAF + Qt-dispatch hazard) |
| `XNodeBrowser::process` (QTimer) | caught + 10-tick backoff (was the retry engine) |
| driver threads (`execute_internal`) | caught → thread exits; no loop |
| Ruby (`evalProtect`) / Python (`mainthread_callback`, pybind) | caught / marshalled to script exceptions |
| graph dump XThread (`graphntoolbox`, UI_DEFERRABLE) | **was uncaught → terminate; now caught, dump lost with a message** |
| paint handlers, menus, stray timers | **now backstopped by `KameApplication::notify`** — a last-resort catch at the Qt event boundary, since Qt does not support exceptions crossing dispatch |

No auto-retry-on-timeout loop remains; every landing site either consumes the
failure or backs off.

## The 2026-07-31 freeze investigation: what the lab settled and what it could not

Field: with PNR on, MCP traffic at 30–47 writes/s and idle pollers, a
privilege-holding transaction on the acquisition thread pins every negotiator
for 33+ s; one episode self-recovered at 11 s with NO starvation message.
A parallel assistant session attributed it to "one 35 s PNR call"; measured
here (solver extracted standalone, M4, -O2, IC early-stop active):

    n=16k: 11 ms   n=32k: 19 ms   n=128k: 99 ms   n=1M: 1.2 s
    n=4M: 5.1 s    n=16M: 22.5 s  n=64M: 108 s

The wave was ~30 k points → ~19 ms/call: a single 35 s call is off by three
orders (it would need a ≥30 M-point wave; a briefly reported "50 M" retracted
to "30 k" flipped the verdict twice — first-hand parameters before theories).
The loop caps the user remembered are real: 32 IC-gated outer iterations,
10 inner.

Harness reproduction with the exact field parameters (22 ms closure,
40 writes/s, 3 idle pollers, 30/s root snapshots, up to 384 nodes):

  * a fresh writer INSIDE the analyze scope is throttled from 40/s to ~4/s —
    fresh commits on a bundled subtree DO negotiate (unbundle path) and DO
    respect privilege.  The "fresh ops bypass fair-mode" asymmetry applies
    only to paths that never need an unbundle;
  * the field COUPLING is the shared entries list: when the analyze
    transaction also writes its scalar entry (root-scope commit spanning its
    subtree + the shared list), victims degrade ×50 (max gap 9 ms → 459 ms)
    and the MCP-like writer is throttled to ~15 %, while the analyze itself
    stays healthy (1.1 closure runs per commit);
  * but 33 s was NOT reached at any size tried — the quantitative pin needs
    an ingredient outside the pure-STM harness (interface mutexes interleaved
    with negotiation, the real listener topology, main-thread event-loop
    granularity...).  Decisive artifact: the field is reproducible on demand
    now, so `sample kame 5` DURING the pin (not the post-mortem .ips) will
    name the holder and its blockage directly.

Also explained from code: 33 s of pinning without the HANG abort is expected —
`_hang_hits` is local to one `_negotiate_internal` call, so only a thread that
sits in ONE call for 3 x 5 s caps aborts; threads cycling in and out of
negotiation can be pinned indefinitely without tripping it.  And the 11 s
recovery without a starvation message means the bound did not fire there
(lucky gap instead); MIN_RETRIES=8 with 5 s sleep caps can defer the bound
past any realistic freeze — the proposed MIN_RETRIES=2 remains open, as does
the release-default KAME_STM_HANG_ABORT_N=0 (the 11 s self-recovery was 4 s
short of today's abort).

## The verdict: KAME retires STM-HIGHEST (user, 2026-07-31)

The arc's measurements reached their terminus. Each tier contract is sound in
isolation — HIGHEST never waits; NORMAL privilege never expires (completion
guarantee); revocable tiers time out — but their **meeting point** is a
structural hole:

    HIGHEST's fair-mode immunity is its defining contract,
    so it is the ONE contender privilege cannot stop.
    When closure_time x HIGHEST_rate >= 1 on a shared linkage,
    the privilege holder resonates into quasi-starvation
    while its privilege pins every other negotiator.

Lab (field parameters, 22 ms closure, shared entries list): adding a 50 /s
HIGHEST commit stream took the analysis transaction from 1.1 to **15.5
closure re-runs per commit** (Rule 0 acquitted: 1 strip per 30 s).  Field:
"PNR ON alone hangs it, OFF recovers" — the ON action itself starts a ~20 ms
closure racing the record stream; every freeze, recovery and abort of
2026-07-30/31 fits this one mechanism.  No bounded arbitration can bridge it:
letting HIGHEST wait for the holder breaks HIGHEST's bound; expiring the
holder breaks NORMAL's completion guarantee (both already ruled out).

So `AcquisitionPriority` now grants only the **OS-level** elevation (CPU
preference is a thread property with no fair-mode immunity), and the STM tier
of the acquisition loop is NORMAL again.  What stays, and why:

* the kamestm HIGHEST tier, Rule 0, the side word, the priority tests — the
  machinery is correct for hosts honouring the deployment precondition
  `HIGHEST_rate x longest_peer_closure << 1`; KAME with per-record analyses
  cannot;
* `ScopedDemoteRealtime` sites — armed only at HIGHEST, now no-ops that
  document intent and re-arm if a future deployment restores the tier;
* the 20 ms downstream budget, the starvation bound (10 s), the exemptions
  and nets — all priority-independent;
* the OS-priority split (thread property vs transaction property), which this
  verdict retroactively justifies: the two were never the same thing.

With fair-mode effective against ALL contenders again, the long-closure
holder completes promptly (lab: 1.1 re-runs), freezes end in well under a
second, and the watchdog/starvation-bound tuning questions lose their
urgency (defaults left as shipped).

## The budget was the second immunity — the wait behind privilege is now exempt

With STM-HIGHEST retired, the field still froze under PNR (user: 「PNRだとまだ
ダメです。budgetのせい？」).  Correct: the wait budget's expiry escape was
deliberately not gated on fair-mode ("returning is not barging — the caller's
CAS loses to a committing holder like any other"), a rationale that assumed
µs holders.  A 20 ms-closure privilege holder breaks it: every primary
driver's record path carries the 20 ms budget, so any driver fair-blocked
longer than that became a **fair-mode-immune spinner — the exact disease that
retired HIGHEST the same day**, re-invalidating the holder each closure while
honest negotiators pinned behind its privilege.

Field-parameter A/B in the harness (no HIGHEST anywhere):

                        analyze re-runs   HANG dumps   longest pin
    budgets on (KAME)        2.3             372         12.5+ s   ← the field freeze
    budgets off              1.1               0         none
    budgets on + fix         2.2               0         none

The fix: the loop-top and tail budget escapes are gated on
`fair_mode_blocks_me`, and the budget's sleep clamps are suspended for the
round while fair-blocked (else the expired thread busy-spins through
zero-length waits instead of waiting).  The principle, now stated once for
both incidents: **privilege is the completion guarantee, and nothing may be
immune to it** — not a priority tier, not a budget.  The budget still bounds
every other wait (the fixed arm's writers pass 425 vs 83 unbudgeted), and
expired-lowprio stamps still unblock, so a dead holder cannot pin a budgeted
thread.  A record can now be late by one holder's closure; it is never lost,
and the system never freezes for it.

## The watchdog reports; it no longer kills (release)

`KAME_STM_HANG_ABORT_N` release default 3 → 0 (user, after the arc's root
causes landed).  The abort was tuned for true deadlocks and instead executed
recoverable states twice in the field: an 11 s self-recovery had a 4 s margin
on it, and a live holder grinding 33+ s — contract-legitimate waiting — took
the unsaved measurement with it.  With ghosts structurally prevented and both
fair-mode immunities gone, the remaining >15 s waits are live-holder waits;
the [HANG] dumps keep naming the blocker, the starvation bound frees the UI
tiers, and a genuine deadlock is the operator's call after saving.  Debug
builds keep 3 — there the core dump is the point.

Residual corner, noted: with no abort, a lowprio thread already sunk in 5 s
sleep caps accrues retries at ≥5 s each, so its starvation exit can lag to
~40 s (MIN_RETRIES=8).  MIN_RETRIES=2 would cap that at ~bound+ε; still open.
