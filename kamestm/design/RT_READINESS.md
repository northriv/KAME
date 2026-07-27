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

### The open question, stated precisely

Age-based promotion exists: a transaction older than
`KAME_STM_PRIV_AGE_NORMAL_US` (**300 µs** on macOS/Linux; 10 ms on Windows for
the scheduler quantum) becomes eligible for privilege. **The measured tail is
70–200× that threshold.** So one of the following is true, and which one is not
yet diagnosed:

1. promotion is never reached — the transaction's age never accumulates,
   e.g. if the attempt's start stamp is refreshed on retry, so a repeatedly
   losing transaction stays permanently "young";
2. promotion is reached but the promoted transaction still loses the slot;
3. promotion is granted and effective, but the *number* of rounds before it is
   large enough that 300 µs × rounds reaches tens of ms.

(1) touches a known fidelity item: the TLA+ liveness argument ranks by an
*iteration counter* that only increases, while the C++ ranks by a start
timestamp — recorded in the pre-submission dossier as examined and argued sound
via the global minimum, but not isomorphic. The measurement does not settle
that argument; it does say the observed waiting is two orders of magnitude past
the point where promotion was supposed to bite.

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

    S1'  Diagnose which of (1)/(2)/(3) above produces the tail.  Cheap:
         instrument rounds-before-promotion in a throwaway build (the existing
         KAME_ADAPT_INSTRUMENT is not throughput-neutral, so it must not share
         a run with the latency numbers above).
    S2   Bound per-commit work (bundle churn is O(subtree)), which any
         waiting bound is stated in terms of.
    S3   Whatever S1' finds: make NORMAL's waiting bounded rather than merely
         eventual, as a change to the arbitration itself — not a fast-path
         branch and not a privileged bypass.
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
