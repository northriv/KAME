# kamestm

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20(MinGW%20%2B%20MSVC)-lightgrey)]()

Lock-free software transactional memory (STM) primitives — the
snapshot/transaction core from the [KAME](https://github.com/northriv/KAME)
measurement framework, extracted as a stand-alone, **header-only**
library plus three small `.cpp` (`threadlocal` / `xthread` / `xtime`).

Dual-licensed under your choice of **Apache 2.0 OR GPL-2.0-or-later**
so it embeds cleanly into permissive / proprietary projects (Apache
path) or links into GPLv2-only projects such as KAME itself (GPL path).

**Production-stable in KAME since 2008** — the STM core has been the
foundation of the KAME node tree under 24/7 research-lab operation on
every release from that year onwards.  Builds and passes the standalone
test suite on macOS clang, Linux gcc/clang (64-bit + 32-bit),
Windows MinGW64 + lld, and Windows MSVC.

## What's in here

The library provides the **snapshot + transaction-commit core** of the KAME
STM design.  It builds on the **atomic primitives — `atomic.h`,
`atomic_mfence.h`, and `atomic_smart_ptr.h` (the tagged-pointer lock-free
`atomic_shared_ptr` / `local_shared_ptr` that is the engine under every
Snapshot) — which live in [`kamepoolalloc/`](../kamepoolalloc)**, their single
shared home (shared with the pool allocator), and are included header-only here
(no `libkamepoolalloc` link; see [Dependencies](#dependencies)).

| Header | Role |
|---|---|
| `atomic_queue.h` | Lock-free MPMC queue |
| `xthread.h` + `xthread.cpp` | `XMutex` / `XCondition` / `XRecursiveMutex` wrappers around `std::mutex` |
| `xwaitcell.h` | `XWaitCell` — timed wait-on-address, the primitive a losing transaction parks on. **Mutex-less** on macOS (`__ulock_wait`) and Linux (`futex(FUTEX_WAIT_PRIVATE)`); portable mutex + condvar fallback elsewhere (see [Realtime behaviour](#realtime-behaviour)) |
| `threadlocal.h` + `threadlocal.cpp` | `XThreadLocal<T, Tag>` with deterministic per-thread teardown |
| `xtime.h` + `xtime.cpp` | Monotonic time helpers used by Lamport-clock serial numbers |
| `transaction.h`, `transaction_definitions.h`, `transaction_impl.h`, `transaction_signal.h` | The STM core: `Snapshot<XN>`, `Transaction<XN>`, `Node<XN>`, `Talker<...>` |
| `transaction_negotiation.h`, `transaction_neg_impl.h` | Negotiated retries (adaptive backoff used by the `iterate_commit` family) |

Out of scope (lives in `kame/` proper): `XNode`, the higher-level
node hierarchy on top of `Transactional::Node`.  Pull-out of that
layer is tracked separately.

## The STM model

KAME's core data model is a lock-free, snapshot-based STM (`transaction.h`).
All instrument data lives in a tree of `Node<XN>` objects; reads and writes are
expressed as **snapshots** and **transactions** rather than locks.

```
Node<XN>
 └─ Linkage  ──atomic_shared_ptr──▶  PacketWrapper
                                          └─ Packet
                                              ├─ Payload   (user data)
                                              └─ PacketList (child packets)
```

**Reading — O(1) snapshot:**

```cpp
Snapshot<NodeA> shot(node);         // atomic load, no lock
double x = shot[node].m_x;
```

**Writing — optimistic transaction with automatic retry:**

```cpp
node.iterate_commit([](Transaction<NodeA> &tr) {
    tr[node].m_x += 1;             // copy-on-write on first access
});                                 // retried automatically on conflict
```

**How commits work:**

1. `Transaction` saves `m_oldpacket` at construction.
2. `operator[]` clones the payload (copy-on-write) on first write, stamping it with a unique serial.
3. `commit()` does a single CAS on `Linkage`; if `packet != m_oldpacket` a conflict is detected and the transaction retries.
4. Listeners receive deferred events only after a successful commit — no intermediate states are visible.

## Lock-free atomic shared pointer

The O(1) snapshot reads and CAS-based commits above require a shared pointer that is itself lock-free. `atomic_shared_ptr` (in `kamepoolalloc/atomic_smart_ptr.h`, introduced in January 2006 as part of the 2.0-beta3 rewrite) provides this. It is a custom implementation of what C++20 calls `std::atomic<shared_ptr>`.

The core technique embeds a small **local reference counter** in the low bits of the pointer to the reference-control block — bits guaranteed zero by allocator alignment. `acquire_tag_ref_()` atomically increments this local counter via CAS to "pin" the pointer for reading; `release_tag_ref_()` decrements it. Between these two calls, even if another thread swaps the pointer, the object cannot be freed because the local count is non-zero. A separate **global reference counter** in the control block tracks long-lived ownership (copies held across scopes). Setters transfer any outstanding local count to the global counter before swapping, so `release_tag_ref_()` can fall back to decrementing the global counter if the pointer changed.

For types that inherit `atomic_countable` (notably `Payload`), the global reference counter is stored inside the object itself (**intrusive counting**), eliminating a separate heap allocation per shared-pointer instance. Non-intrusive types get an external control block (`atomic_shared_ptr_gref_`).

**Comparison with standard-library implementations (as of late 2024):**

| Implementation | Technique | Lock-free? |
|---|---|---|
| libstdc++ (GCC) | Spinlock on internal table | No — vulnerable to priority inversion |
| MSVC | Lock bit + `WaitOnAddress` | No — blocking under contention |
| libc++ (Clang) | Not yet implemented | N/A |
| KAME (2006–) | Tagged-pointer CAS | Yes — lock-free reads and writes |

The CAS primitives and memory barriers delegate to `std::atomic` and `std::atomic_thread_fence` (`kamepoolalloc/atomic.h` / `atomic_mfence.h`). The earlier hand-written x86 / PowerPC / ARM assembly fences have been removed in favour of this portable C++17 path.

**Multi-node consistency** is achieved through a *bundling* protocol: a parent packet absorbs child packets via multi-phase CAS protocol, making the entire subtree consistent under a single atomic pointer. A `m_missing` flag marks packets with stale children, driving re-bundling on demand.

**Collision negotiation (livelock-free, priority/age-ordered):** when two
transactions repeatedly collide, the negotiate machinery
(`ScopedNegotiateLinkage::_negotiate()`) lets a single *oldest* transaction
win rather than letting them busy-retry against each other. Each transaction
carries a fixed `m_started_time` tidstamp (start time packed with its thread
id, never re-stamped across retries); on contention it tags each contended
linkage's own `m_transaction_started_time` slot via
`Snapshot::tag_as_contender()` under an **oldest-wins** rule (older = earlier
start), with a symmetric ~100 µs `KAME_STM_PREEMPT_WINDOW_US` burst window
damping preemption between near-contemporaneous threads. A transaction that
keeps losing escalates its tag to a *privileged* (Reserved-kind) stamp once
the livelock probe fires (eligibility keyed on tag-ownership + retry count,
not wall-clock age); only such Reserved stamps hard-block a peer's CAS
(`fair_mode_blocks_me`) — a plain tag merely shortens the loser's adaptive
backoff. Priority bands modulate expiry: only LOW-priority holders (LOWEST /
UI_DEFERRABLE / SCRIPTING) can be expired or evicted; NORMAL / HIGHEST
(measurement / driver-critical) are immune. Tags are released by
`drop_tags_n_privilege()` (a CAS-based mine-only clear) at commit success, at
`~Transaction()` (abort / RAII), and at standalone-`Snapshot` completion.
Non-privileged contenders **park** (adaptive backoff, then a timed
wait-on-address — `XWaitCell`, mutex-less on macOS and Linux; see
[Realtime behaviour](#realtime-behaviour)) instead of spinning, so the
oldest / highest-priority transaction
always makes progress — model-checked livelock-free in TLA+ (the Layer-2
`BundleUnbundle_*_LLfree` specs below model this per-linkage tag as a
per-node `priorityTag`; see [tests/VERIFICATION.md](tests/VERIFICATION.md) §3
— exhaustive for the checked thread counts and tree shapes). The global
`s_privileged_tidstamp` / `try_register_privileged_tidstamp` slot is the
`KAME_PER_LINKAGE_PRIVILEGE=0` fallback, compiled out by default. This
replaces the earlier proportional-timestamp-wait backoff.

`iterate_commit_while(lambda)` lets the caller abort the retry loop (return `false` from the lambda to stop), enabling conditional transactions.

> **Caution:** Taking a nested `Snapshot` inside a transaction can trigger bundling, which may cause the transaction's CAS to always fail. This is not a data corruption issue but a liveness issue — the transaction retries indefinitely. This occurs when the `Snapshot` target is an ancestor of the transaction target, or when hard links exist (a child with two parents) and a `Snapshot` on one parent's tree interferes with the other. Use `tr[*node]` instead of a nested `Snapshot` in these situations.
>
> The hard-link case is formally modelled in `tests/tlaplus/BundleUnbundle_hardlink_*.tla` (seven topology/pattern variants, incl. the conditional nested-sub-bundle gate-scope model); see `tests/VERIFICATION.md` §5.

## Comparison with other STM designs

*The following comparison was written by Claude (Anthropic) based on analysis of the source code.*

Most widely-used STMs (GHC/Haskell `TVar`, Clojure `Ref`/`dosync`, ScalaSTM) are **flat**: the unit of transaction is a set of independent transactional variables. KAME's STM is instead **tree-structured** — the entire instrument node tree is the shared state, and snapshots are always subtree-consistent. This difference drives several design choices:

| Aspect | Flat STMs (Haskell, Clojure, ScalaSTM) | KAME STM |
|---|---|---|
| Conflict granularity | Per-variable | Per-packet (subtree root) |
| Read model | `readTVar` / `deref` inside transaction | `Snapshot` (outside) or `tr[*node]` (inside) |
| Consistency scope | The read/write set actually accessed, tracked dynamically (not declared up front) | Entire subtree, guaranteed by bundling |
| Commit log | Redo log or write set | Copy-on-write + CAS on single `Linkage` |
| Retry primitive | `retry` / `orElse` (Haskell) | `iterate_commit` / `iterate_commit_while` |
| Blocking | `retry` suspends on read-set change | No data-structure locks; a repeatedly-colliding Tx yields/parks to the privileged (oldest / highest-priority) Tx |
| Memory management | GC | Lock-free `atomic_shared_ptr` (ref-counted) |
| Hard real-time suitability | Limited (GC pauses) | No GC pauses, and a declared wait budget converts the tail into a chosen number — **measured** on a `PREEMPT_RT` host, MAX = budget + ~200 µs, 38.3 M commits with zero over 3 ms at a 1 ms budget ([below](#realtime-behaviour)). Still not hard-RT in a strict WCET sense: CAS retry *counts* are not bounded, and the budget cannot bound the wait behind a live privilege holder — that one is the deployment's to bound, with core isolation |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to age-ordered privileged-Tx negotiation (the colliding losers yield to the oldest transaction) rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** Both use a global version clock and keep a read/write log per transaction, but differ on per-object metadata — TinySTM uses per-object version locks, whereas NOrec deliberately keeps *none* (it validates the read set by value against the global clock; the name is "No Ownership Records"). KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

## Realtime behaviour

The STM has **no WCET bound to offer**: CAS retry counts are not bounded, and
the design deliberately does not bound how long a privilege holder takes to
finish.  What it has instead is a measurement, under the role mix an
instrument-control deployment actually runs, on a `PREEMPT_RT` host, quoted
against that host's own floor.

### "Unbounded retries" and "model-checked livelock-free" are both true

The precise word for what is proven is **starvation-freedom**, not bounded
wait-freedom, and the difference is narrower than "not bounded" makes it sound.
Taking the concessions first, because both are real:

* **Every terminating execution retries finitely often.**  That is what
  `<>AllDone` says, and on a machine of finite speed it is true by
  construction.  Divergence is not on the table.
* **For each checked configuration a bound even exists.**  The state space is
  finite and contains no progress-free lasso, so any progress-free stretch is
  shorter than the state count and the total is that times the finite number of
  progress events.  "Unbounded" is not a statement about the model.

What is missing is a bound for the **deployed** system, and two things stand in
the way:

* **The checking is per-configuration** — three threads, two children,
  `MaxCommits = 1` — with no ∀-thread result, so nothing transfers to a tree of
  dozens of drivers.  And the number would not be usable if it did: it is in
  TLA+ steps over ~10⁸–10⁹ states, and a step is not a CAS attempt.
* **The models drain, and a deployment does not.**  `Threads` is a fixed set,
  each thread holds a finite `iterBudget`, and `AllDone` *is* "every budget
  exhausted" — so a retrying transaction is guaranteed to run out of
  competitors.  The quantity an instrument cares about is retries against a
  *continuing* arrival stream, which the specs do not represent at all.  This
  is the load-bearing gap, not the thread count.

So read "not bounded" as **no bound is established for the deployed
configuration**, not as "retries can diverge".

What livelock-freedom does buy is real, and is exactly what Layer 1 lacks — its
`<>AllDone` check fails with a lasso, which is why Layer 2 exists.  A
transaction that keeps losing escalates to a privileged stamp its peers must
yield to, including first-attempt peers (`retry == 0` still checks
`fair_mode_blocks_me`).  The escalation is probe-gated, so the guarantee is
about *what eventually happens* rather than *when* — the textbook gap between
starvation-freedom and a constant.

For scale, measured rather than argued: **1.002 attempts per commit over
38.3 M commits, 1.45 on the slow ones, maximum 5.**

### What was measured

All of it comes from `transaction_priority_mixed_test`, which times the
acquisition thread's record commit — the deadline-bearing half of an
acquisition cycle — while NORMAL driver peers, a UI thread taking root
Snapshots and a SCRIPTING thread contend against it.  Host throughout: Ubuntu
26.04 `7.0.0-29-realtime` (`CONFIG_PREEMPT_RT=y`), i5-7500, cores 2–3 under
`isolcpus`/`nohz_full`/`rcu_nocbs`, IRQs steered to 0–1, `performance`
governor.  **Quote every number below against the host's own floor of 17 µs**
(`rtla osnoise`, 120 s, Max Single — C-states, SMIs and `nohz_full` wake-ups in
one number); an absolute latency is a property of the machine, which is why
`KAME_MIX_DEADLINE_US` turns MAX into a pass/fail assertion only when asked.

### Two tiers, and the difference is not a matter of degree

`Priority::HIGHEST` leaves the negotiator's round loop before it can sleep —
`if(entry_pr == Priority::HIGHEST) break;` sits at the top of the loop, above
both `negotiate_sleep` call sites — so a HIGHEST commit never parks.  NORMAL
does, in 1–2 ms chunks.  Same host, same roles, only the tier (120 s):

| tier | p50 | p99 | p99.9 | **MAX** |
|---|---|---|---|---|
| HIGHEST (the library's ceiling) | 768 ns | 2.05 µs | 20.5 µs | **95.1 µs** |
| NORMAL, 20 ms budget | 768 ns | 1.28 µs | 3.67 ms | **20.15 ms** |

The median is identical and the tail is 200× apart.  "It kept up on average"
was never the question.  The signature is unambiguous: the other roles
completed a mean of 2,004 commits during each slow NORMAL commit against 13 in
the HIGHEST arm — a thread asleep while the system works, the same shape
`transaction_latency_bench`'s four-symmetric-thread control shows when it
reaches 3.1 ms at p99.99 with a 32.6 ms max.

`SCHED_FIFO` changes none of it (43.8 k/s and MAX 20.15 ms with it, 43.3 k/s
and 20.19 ms without): `negotiate_sleep` is a **voluntary** wait, and no
scheduling class shortens one.

HIGHEST's number carries a precondition the table cannot show, because the
measurement satisfies it: **HIGHEST commit rate × longest peer closure ≪ 1.**
Never parking cuts both ways — a HIGHEST commit is also immune to fair-mode,
so it does not yield to a *privileged* peer either, and each of its commits
that lands inside that peer's closure invalidates the whole closure.  With
µs-scale closures (this table) the collision window is negligible; when a
peer runs ms-scale closures past the meeting point, the peer's privilege
stops converging — reproduced at 22 ms closures against a flat-out HIGHEST
churner as 1.1 → 15.5 closure re-runs per commit, a privilege holder pinned
not by the scheduler but by arithmetic.  Everything waiting behind that
holder waits through every re-run, exempt from any budget.  KAME itself now
runs acquisition at NORMAL with OS-level elevation only, for exactly this
reason: its analysis closures are ms-scale.  Use HIGHEST where the
precondition is a property of the design, not a hope about the load.

### At NORMAL the wait budget is the only bound — and it delivers its value

`ScopedWaitBudget` (`XPrimaryDriver::downstreamWaitBudgetUS()`, default 20 ms)
is inert at HIGHEST and binding at NORMAL.  Swept with the acquisition thread
at `SCHED_FIFO` on the isolated core and every other thread together on the
housekeeping core, 60 s each:

| budget | commits/s | mean | **MAX** | MAX − budget | clipped |
|---|---|---|---|---|---|
| 2 ms | 76,904 | 7.99 µs | 2.179 ms | 179 µs | 0.333 % |
| 1 ms | 131,721 | 4.10 µs | 1.223 ms | 223 µs | 0.320 % |
| 500 µs | 185,700 | 2.41 µs | 0.662 ms | 162 µs | 0.304 % |
| 200 µs | 251,933 | 1.53 µs | **0.408 ms** | 208 µs | 0.334 % |

**MAX = budget + a constant ~200 µs, with no floor down to 200 µs** — and
throughput *rises* 3.3× as the budget falls, because a clipped commit stops
sleeping and retries.  The clip rate is invariant at ~0.32 %: the same
population of commits is caught, just earlier and more cheaply.  (The 4.7 %
throughput cost documented for the 20 ms default was measured on a different
arm and does not hold here; here smaller is better on both axes.)

Confirmed at length — 300 s, `SCHED_FIFO` + isolation, 1 ms budget:
**38,303,308 commits, MAX 1.288 ms, zero over a 3 ms deadline**, with every
other role healthy (UI 42.3 k/s, SCRIPTING 129.7 k/s, NORMAL 100.6 k/s).

### Isolation is what makes the budget work, and FIFO without it is worse than nothing

The budget bounds every wait *except* the one behind a live privileged peer,
which is contractually exempt — so that one's length is the holder's
scheduling delay, plus the holder's closure re-runs if a fair-mode-immune
contender keeps colliding with it (the HIGHEST precondition above; absent a
HIGHEST role only the scheduling term remains, which is the case measured
here).  It shows up exactly there:

* **Unpinned**, MAX sticks at **12–13 ms for every budget from 5 ms down to
  500 µs** while the clipped count saturates.  The budget is not what is being
  measured any more.
* **Pinned** — acquisition alone on the isolated core, every contender together
  on the housekeeping core — a holder is always promptly scheduled among its
  peers and the exempt residue vanishes (the table above).
* **`SCHED_FIFO` without isolation FAILS.**  Only the acquisition thread is
  elevated, so it preempts the very CFS holders it then waits behind: UI fell
  to 144 commits/s and SCRIPTING to 176 (from 42.3 k and 129.7 k), both flagged
  by the livelock watchdog at 6,001 ms, while the acquisition thread ran away
  at 337 k/s and *still* took 50.9 ms on its own worst commit.  A textbook
  priority inversion.  **FIFO and isolation ship together or neither ships.**

Also refuted, since it was the obvious suspect: the cross-subtree
`XSecondaryDriver` role is not what the 12–13 ms residue is made of.  Turning
it off halves the clipped population and leaves MAX where it was.  Narrowing
that scope is worth doing for throughput; it does not buy the tail.

### What the STM does not bound

Each of these is a design decision rather than a gap, and a realtime deployment
has to supply the missing bound itself:

* **A privilege holder's completion time.**  NORMAL and HIGHEST privilege
  never expire — that immunity *is* the completion guarantee — and the wait
  behind a live privilege is exempt from the wait budget.  Two things can
  stretch it, and the STM bounds neither: the holder's **scheduling delay**
  (if the OS does not run the holder, nothing in the STM rescues the waiter —
  the bound the section above measures from both sides: supply isolation and
  the exempt wait disappears, withhold it and no budget can reach the tail),
  and the holder's **closure re-runs under a fair-mode-immune contender**
  (the HIGHEST rate precondition — isolation does not help there, since it is
  not a scheduling problem).  (Only the LOW band — LOWEST / UI_DEFERRABLE /
  SCRIPTING — can be expired or evicted.)
* **Transaction scope**, the dominant *throughput* term and the caller's to
  choose.  Measured 2×2 at HIGHEST on the RT host, acquisition commits/s:
  neither 146.9k · `SCHED_FIFO` + pinning only 155.0k · one NORMAL peer whose
  scope spans the acquiring driver's subtree (the `XSecondaryDriver` role)
  89.4k · both 57.8k.  FIFO and pinning cost nothing here (+6 %); the
  cross-subtree peer costs 1.64× on its own, and the pair is super-additive.
  It does not follow through to latency — see the refutation above.
* **Allocation.**  Every commit clones a payload, so the allocator sits on the
  deadline path and its preconditions are inherited — in particular
  [`kamepoolalloc`](../kamepoolalloc)'s `kame_pool_prewarm()`, called from the
  realtime thread before the time-critical section.  Skipping it cost the
  measurement above a **~400 µs first commit** (the pool's freelist pre-fill
  faulting five size classes' first chunks at once), immovable across every
  other knob until the precondition was honoured.

### The configuration that follows

Everything above collapses into four requirements.  They are not independent —
each of the first three is what makes the next one mean anything:

1. **Isolate the deadline-bearing thread** (`isolcpus`) and put **every other
   STM thread together** on the housekeeping cores.  Not for cache or for
   tick-freedom: so that a privilege holder is always promptly scheduled, since
   the wait behind one is the bound the budget cannot reach.
2. **`SCHED_FIFO` only on top of (1).**  On its own it is a priority inversion
   generator, and it buys nothing measurable even when correct.
3. **A wait budget sized to the deadline.**  MAX lands at budget + ~200 µs, so
   pick the budget and read off the guarantee.  Smaller is better on both axes
   here, so size it from the deadline rather than from a throughput fear.
4. **Prewarm the allocator from that thread** before the time-critical section
   (see below), or pay ~400 µs on the first commit.

Measured end to end at 1 ms: **38.3 M commits, MAX 1.288 ms, zero over 3 ms.**

### No lock on the negotiation route

A losing transaction parks on `XWaitCell` (`xwaitcell.h`) rather than spinning,
and on macOS and Linux that park is **mutex-less** — `__ulock_wait` and
`futex(FUTEX_WAIT_PRIVATE)` respectively, the kernel's value-compare on a
generation word closing the lost-wakeup window a condition variable would need
a mutex for.  This matters only under a scheduler that enforces priority: the
fallback's `std::mutex` is a plain `pthread_mutex` with no priority
inheritance, so a high-priority committer can be made to wait on a preempted
low-priority one — bounded, since the block itself yields, but unbounded once a
medium-priority thread interposes.  Removing the mutex removes the question;
`PTHREAD_PRIO_INHERIT` would only have bounded it.

It is **not** a throughput change and does not claim to be: interleaved against
a forced-fallback build it measures identical in commits/s and in p50/p99/p99.9,
because the sleep path is reached in 0.0001–0.05 % of commits.  Force the
fallback with `-DKAME_XWAITCELL_ULOCK=0` / `-DKAME_XWAITCELL_FUTEX=0`;
`xwaitcell_test` passes on all three backends.

## Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC:

- **Layer 1 — `atomic_shared_ptr`:** tagged-pointer CAS protocol with local/global reference counting, drain release, and `scoped_atomic_view` ([spec](tests/tlaplus/atomic_shared_ptr.tla)). Safety only — the bare primitive is intentionally *not* livelock-free.
- **Layer 2 — bundle/unbundle + commit:** 2-/3-level subtree bundling with a livelock-free privileged-TID negotiate mechanism, static and dynamic (online insert/release) ([2-level](tests/tlaplus/BundleUnbundle_2level_LLfree.tla), [3-level](tests/tlaplus/BundleUnbundle_3level_LLfree.tla), [dynamic](tests/tlaplus/BundleUnbundle_2level_LLfree_dynamic.tla)). Exhaustively model-checked **safe + livelock-free** without `CONSTRAINT` (the LL-free design makes the state space naturally finite — no artificial bound); the largest single exhaustive run reaches **~641 M distinct states** (3-level all-root, 15 h on the ISSP ohtaka supercomputer), over a billion across the LL-free configurations combined. (Raw state counts are **spec-version-specific** — they shift as the spec evolves, so cross-version comparison isn't meaningful; see [tests/VERIFICATION.md](tests/VERIFICATION.md) §3–§4 for the current-spec figures.) These are exhaustive results for the checked configurations (fixed thread counts and tree shapes), not an unbounded ∀-thread proof. The property is `<>AllDone` over a *draining* workload, so it is starvation-freedom, **not** a bound on retry counts — [Realtime behaviour](#unbounded-retries-and-model-checked-livelock-free-are-both-true) sets the two side by side.
- **Hard-link topologies:** multi-parent / one-child races that reproduce and fix a production abort via a Phase-4 reachability gate and a Phase-3 skip-Null fix (`tests/tlaplus/BundleUnbundle_hardlink_*.tla`).

**Slide decks** — start at the **coverage overview** ([EN](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_overview_en.html) · [JA](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc_ja/slides_overview.html)), a hub linking every layer with a full coverage matrix. Individual decks (each with a Japanese counterpart under `doc_ja/`): [Layer 1](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer1_en.html), [Layer 2 base](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_en.html), [Layer 2 LLfree](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree.html), [3-level](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree_3level_en.html), [dynamic](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree_dynamic_en.html), [hard-link](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_hardlink_en.html).

C11 translations of each layer are verified with [GenMC](https://github.com/MPI-SWS/genmc) under the RC11 memory model: TLA+-derived tests (`tests/tlaplus/test_*.c`) and C++-derived protocol tests (`tests/cds_atomic_shared_ptr/`).

## Dependencies

- C++17 toolchain — gcc 9+, clang 10+, **and MSVC (cl)**.  The standalone
  tests build and pass on macOS clang, Linux gcc/clang
  (64-bit + 32-bit), Windows MinGW64 + lld, and Windows MSVC (cl 19.51).
  Nothing in the library is POSIX-only; the one platform-gated *test*
  feature is `transaction_priority_mixed_test`'s OS-scheduling arm
  (`#if defined(__linux__)`, and `SKIPPED` rather than silently green
  when the process may not set `SCHED_FIFO`).
  The MSVC build needs no opt-in flag: kamestm already used
  `std::atomic` / `thread_local` and carried `_MSC_VER` branches for
  the few primitives (popcount, fences, rdtsc); commit `60cfc7dc`
  added the last portable shim (`ctz_u64` mirroring `popcount_u64`)
  and rewrote the function-local `constexpr` constants nested lambdas
  use as `static constexpr` so MSVC accepts them inside `if constexpr`
  (C2131 / C3493).
- [`kamepoolalloc`](../kamepoolalloc) — sibling library providing
  `Transactional::allocator<T>` and the lock-free pool used by every
  Snapshot allocation.  It is **also the single home of the Layer-0 atomic
  primitives** (`atomic.h` / `atomic_mfence.h` / `atomic_smart_ptr.h` —
  `atomic_shared_ptr` / `local_shared_ptr`), which `transaction.h` includes
  HEADER-ONLY (no `libkamepoolalloc` runtime link is needed for them).  The
  STM core includes
  `kamepoolalloc/allocator.h` via the consumer's INCLUDEPATH; falling
  back to `std::allocator` requires defining `USE_STD_ALLOCATOR`
  before including `transaction.h`.  (`kamepoolalloc`'s own MSVC
  build is default-on — opt OUT via `KAME_DISABLE_POOL_MSVC` — so
  unless explicitly disabled, MSVC users get the live pool here too.)

## Build / Use

This is intended to be consumed from a parent build (KAME itself, or
a downstream user's CMake/qmake project).  Add to your INCLUDEPATH:

```cmake
target_include_directories(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/path/to/kamestm
    ${CMAKE_CURRENT_SOURCE_DIR}/path/to/kamepoolalloc)
```

Compile `kamestm/threadlocal.cpp` + `kamestm/xthread.cpp` +
`kamestm/xtime.cpp` into your target.

A stand-alone `kamestm.pro` / `CMakeLists.txt` producing a
`libkamestm.dylib` is on the roadmap.

## Tests

Built by the `tests/` CMake scaffold and run with `ctest`
(`cmake -S tests -B build && cmake --build build && ctest --test-dir build`):
**19 registered tests**, plus three drivers that are built but deliberately not
registered because they take command-line arguments and are run on purpose (the
two `*_mixed` throughput drivers and `transaction_latency_bench`).
Five layers, from primitive to whole-protocol:

**Primitives** — the lock-free building blocks, exercised directly:

| test | covers |
|---|---|
| `atomic_shared_ptr_test` | tagged-pointer `local_shared_ptr` load / store / CAS / swap under contention |
| `atomic_scoped_ptr_test` | single-owner scoped pointer + `local_weak_ptr` promotion |
| `atomic_queue_test` | lock-free MPMC queue |
| `mutex_test` | the `std::mutex` / `shared_mutex` wrappers |
| `xwaitcell_test` | the timed wait-on-address primitive negotiation parks on — the ordinary timeout, `usec == 0` meaning poll rather than forever, the lost-wakeup window, a real cross-thread wake, and eight sleepers none stranded. Passes on all three backends, so a compile-time backend choice cannot drift unnoticed |
| `fast_vector_test` | union discipline of `fast_vector<T,N>` — the inline array and the heap vector are union'd, so any method reaching for the inactive member is UB |

**STM functional** — concurrent transactions on the node tree:

| test | covers |
|---|---|
| `transaction_test` | simultaneous transactions on tree-structured objects |
| `transaction_dynamic_node_test` | transactions that **insert / remove / swap** node links concurrently |
| `transaction_negotiation_test` | transactions of *different periodicities* — the slow loop never commits unless the fast loop yields to it via the privileged-Tx negotiation (`ScopedNegotiateLinkage::_negotiate()`: the older/starved Tx escalates to a privileged Reserved tag and the fast loop parks) |

**Payload-integrity stress** — Synchrobench-style mixed-contention throughput
drivers that fill every payload with a per-writer **sentinel** and re-check it
on each read, so any torn / lost / stale commit is caught immediately:

| test | shape |
|---|---|
| `transaction_payload_integrity_test` | single node |
| `transaction_payload_integrity_mixed_test` | mixed read/write contention |
| `transaction_payload_integrity_3level_test` | `Grand → Parent → Child[N]` (one leaf per thread) |
| `transaction_payload_integrity_3level_mixed_test` | 3-level + a tunable fraction of grand-scope (cross-level) commits |

The `3level_mixed` driver takes `seconds threads max_payload cross_ratio` and
reports commits/s; because it is dominated by small per-payload allocations it
also doubles as the STM-workload allocator benchmark (vs `kamepoolalloc`).

**Negotiation, priority and realtime** — who wins a collision, whether the
loser is ever pinned, and how long the winner takes.  The three white-box tests
build with `-fno-access-control` because privilege claims are probe-gated and
cannot be manufactured deterministically through the public API:

| test | covers |
|---|---|
| `transaction_wait_budget_test` | a `ScopedWaitBudget` commit finishes within budget + slack (the slack covers OS scheduling and post-expiry retries — everything the library deliberately does not model) |
| `transaction_starvation_test` | the starvation bound on revocable (LOW-band) priorities; the production default (10 s, `KAME_STM_LOWPRIO_STARVE_MS`, arming after `KAME_STM_LOWPRIO_STARVE_MIN_RETRIES = 2` retries) is exercised by *not* firing in the uncontended arm |
| `transaction_sleep_in_tx_test` | debug-only detector for `msecsleep()` inside a Transaction (built `-UNDEBUG`, or it would pass having checked nothing) |
| `transaction_priv_strip_test` | white-box: `tag_as_contender`'s Rule 0 — HIGHEST strips a stuck foreign non-HIGHEST privilege stamp |
| `transaction_priv_expiry_test` | white-box: the expiry rules on the negotiation predicates themselves, both agreeing consumers |
| `transaction_priv_pin_test` | the behavioural net over the same fix: **no thread may ever be pinned for a watchdog-class stretch**, keeping a 2026-07-30 field crash (SIGABRT via the negotiation HANG watchdog) as a regression |
| `transaction_reanchor_test` | white-box: `newTransactionUsingSnapshotFor` must not orphan planted stamps when it re-anchors the snapshot base |
| `transaction_priority_mixed_test` | the deployment's role mix — HIGHEST acquisition, a budgeted NORMAL downstream, a main-thread UI doing snapshots + structural churn, SCRIPTING — under a stall watchdog (any thread stuck > 5 s = livelock = FAIL). `KAME_MIX_*` add an OS-scheduling arm (`SCHED_FIFO`, pinning from the affinity mask, SCHED_IDLE starvation; Linux, and `SKIPPED` when unprivileged), the cross-subtree `XSecondaryDriver` role, and the record-commit latency distribution with an optional deadline assertion |
| `transaction_latency_bench` | the per-commit latency *tail* (not throughput) under four symmetric threads; not registered, because absolute latencies are machine-specific. Pure observation — it times `iterate_commit` from outside |

**Formal / memory-model verification** — see *Formal verification* above and
[`tests/VERIFICATION.md`](tests/VERIFICATION.md).  GenMC RC11-model-checks both
the C++ `atomic_smart_ptr` implementation directly
([`tests/cds_atomic_shared_ptr/`](tests/cds_atomic_shared_ptr) — `cds_test_*.c`:
load / CAS / multi-CAS / swap / scoped-weak, plus `_excess` / `_noacquire` edge
variants that caught real refcount bugs) and the TLA+-derived C translations of
each protocol layer ([`tests/tlaplus/`](tests/tlaplus) — `test_*.c`).  The TLA+
specs themselves (`atomic_shared_ptr.tla`, `BundleUnbundle*.tla` incl. 2-/3-level,
lock-free, dynamic, and hard-link variants) are checked with TLC.

## License

**Dual-licensed under your choice of EITHER:**

- **Apache License, Version 2.0** — see [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
  Best for embedding into permissive / proprietary projects.
- **GNU GPL, version 2 of the License, or (at your option) any later version**
  — see [LICENSE-GPL-2.0](LICENSE-GPL-2.0).
  Best for linking into GPLv2-only projects such as KAME itself.

Pick whichever license suits your downstream project; see [LICENSE](LICENSE)
for the full dual-grant statement.

Copyright (C) 2008-2026 Kentaro Kitagawa &lt;kitag@issp.u-tokyo.ac.jp&gt;,
The University of Tokyo, ISSP.

Both license grants explicitly require preservation of the copyright
notice and the choice-of-license clause when redistributing this
software, in source or binary form.
