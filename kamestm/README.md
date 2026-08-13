# kamestm

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20(MinGW%20%2B%20MSVC)-lightgrey)]()
[![arXiv](https://img.shields.io/badge/arXiv-2608.12024-b31b1b.svg)](https://arxiv.org/abs/2608.12024)

Lock-free software transactional memory (STM) primitives — the
snapshot/transaction core from the [KAME](https://github.com/northriv/KAME)
measurement framework, extracted as a stand-alone, **predominantly
header-only** library: templates in headers plus three small support
translation units (`threadlocal` / `xthread` / `xtime`).

**Paper:** the design, its 16 years of measurement-platform service and the
formal verification are described in K. Kitagawa, *Formally Verified
Lock-Free Software Transactional Memory for Scientific Measurement*,
[arXiv:2608.12024](https://arxiv.org/abs/2608.12024) (2026) — see
[Citing](#citing).

**What it is for.**  Shared **tree-structured** state with many concurrent
writers and readers, where a reader may hold an immutable subtree snapshot
for as long as it likes — a plot redrawing from one, an analysis pass, a
script inspecting the tree.  It offers O(1) snapshot acquisition, atomic
publication of a whole subtree, and oldest-wins arbitration under
contention, none of which lock the live tree.  If your shared state is a
flat set of independent variables, a conventional STM (or plain atomics)
will serve you better; the tree, the retained snapshots and the
subtree-consistency guarantee are what this one buys.

Dual-licensed under your choice of **Apache 2.0 OR GPL-2.0-or-later**
so it embeds cleanly into permissive / proprietary projects (Apache
path) or links into GPLv2-only projects such as KAME itself (GPL path).

**Production-stable in KAME since 2008** — the STM core has been the
foundation of the KAME node tree under 24/7 research-lab operation on
every release from that year onwards.  Builds and passes the standalone
test suite on macOS clang, Linux gcc/clang (64-bit + 32-bit),
Windows MinGW64 + lld, and Windows MSVC.

**Try it** (builds the 19-test standalone suite; see
[Build / Use](#build--use) to consume the library from your own project):

```bash
cmake -S tests -B build && cmake --build build && ctest --test-dir build
```

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
start) applied *within a priority class* — a validated `HIGHEST` tag is never
overwritten by a lower-priority tagger (Rule 0c: priority sits above age for
the HIGHEST class) — with a symmetric ~100 µs `KAME_STM_PREEMPT_WINDOW_US`
burst window damping preemption between near-contemporaneous threads. A transaction that
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
| Cost of an atomic multi-object commit | Per-variable logging, paid at commit | **p50 ≈ 439 ns + 94.5 ns × nodes**, measured to 17 nodes; free in the tail, where 17× the nodes costs 1.6× the worst case ([below](#realtime-behaviour)) |
| Hard real-time suitability | Limited (GC pauses) | No GC pauses, and a declared wait budget converts the tail into a chosen number — **measured** on a `PREEMPT_RT` host, MAX − budget = 3–7 µs at the shipped 20 ms budget (the host's own floor being 219 ns, measured), 38.3 M commits with zero over 3 ms at a 1 ms budget ([below](#realtime-behaviour)). Still not hard-RT in a strict WCET sense: CAS retry *counts* are not bounded, and the budget cannot bound the wait behind a live privilege holder — that one is the deployment's to bound, with core isolation |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to age-ordered privileged-Tx negotiation (the colliding losers yield to the oldest transaction) rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** Both use a global version clock and keep a read/write log per transaction, but differ on per-object metadata — TinySTM uses per-object version locks, whereas NOrec deliberately keeps *none* (it validates the read set by value against the global clock; the name is "No Ownership Records"). KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

## Realtime behaviour

**A commit's worst-case time is a number you choose.**  Declare a wait budget
(`ScopedWaitBudget`; KAME sets 20 ms from
`XPrimaryDriver::downstreamWaitBudgetUS()`) and every wait inside the commit
is clipped to it: measured MAX − budget is **3–7 µs** from 20 ms down to
1 ms, and a 300 s proof run at a 1 ms budget closed 38.3 M commits with zero
over a 3 ms deadline.  Keep the budget well above the 300 µs deadline-spin
reserve — at or below it the committer never sleeps, and the deferrable
tiers starve (measured −94 % / −98 % at a 200 µs budget).

**Measure the host's floor before quoting any number here**
(`tests/latency_floor`, run under `tests/with_pmqos`): this host's floor
alone spans 67.9 µs (no isolation) → 17.0 µs (`isolcpus`/`nohz_full`) →
**219 ns** (+ PM-QoS), and a figure read against the wrong floor
mis-attributes the machine to the STM.  Everything below is against the
219 ns floor — isolated core, `SCHED_FIFO`, per-thread pinning, the pool's
realtime contract honoured — on a PREEMPT_RT i5-7500, from a plain `Release`
build (**not** `KAME_STM_NEG_DIAG`, whose pass timer puts two clock reads
per bundle inside the path being measured):

| workload | p50 | p99.9 | p99.999 | **MAX** |
|---|---|---|---|---|
| HIGHEST, 5-node commit, peers writing into the same subtree (worst of 3 × 300 s) | 768 ns | 3.58 µs | 10.2 µs | **25.1 µs** |
| the same commit with no peer on its subtree (60 s) | 768 ns | 1.28 µs | 1.28 µs | **1.53 µs** |
| NORMAL under the 20 ms budget (120 s) | 768 ns | 1.05 ms | 10.5 ms | **20.01 ms** |

Reproduce with `tests/transaction_priority_mixed_test` under
`tests/with_pmqos`, `KAME_MIX_OS_FIFO=1 KAME_MIX_OS_PIN=1` inside a
`taskset` onto the isolated pair, at the DEFAULT `KAME_MIX_LEAVES=4` — that
default *is* the 5-node commit, and raising it changes which phenomena occur
at all, not just their size. Rows 2 and 3 add `KAME_MIX_DISJOINT=1` and
`KAME_MIX_ACQ_NORMAL=1`.

Against the previous generation (before the stamp's 2-bit PRIO field, the
commit-lease privilege gate and HIGHEST-vs-HIGHEST spin arbitration) row 1's
stable percentiles improved 13–17 % (p50 896 → 768 ns, p99.9 4.10 → 3.58 µs,
p99.999 12.3 → 10.2 µs) at +3–4 % throughput, and row 3's budgeted tail
improved sharply (p99.9 7.34 → 1.05 ms, p99.999 20.97 → 10.5 ms, MAX still
pinned to the 20 ms budget). Row 1's MAX reads 23.7 → 25.1 µs, which is one
extreme value against another: the three runs behind it are 13.7 / 19.8 /
25.1 µs, so the spread is the statistic. **Row 2 moved the wrong way** —
p50 448 → 768 ns, MAX 1.06 → 1.53 µs on the uncontended path, where none of
the three changes should cost anything (all of them sit in
`_negotiate_internal`, which an uncontended commit never enters). Either the
old row was taken under a configuration that is not written down, or
something on the straight-line path did get slower; unexplained, and not
attributed until bisected.

Three facts a deployment can act on:

* **Composability is cheap and does not set the tail.**  The median is
  linear in the commit's node count — **p50 ≈ 439 ns + 94.5 ns × nodes**,
  measured to 17 nodes within 2 % — while 17× the nodes moves the worst
  case only 1.6×.
* **Contention is the tail, and the lever is topological.**  The first two
  rows differ only in whether peers touch the committed subtree: 16× in
  MAX, with nothing over 1.6 µs in 19 M uncontended commits.  A root-scope
  `Snapshot` or `Transaction` bundles every subtree beneath it — a subtree
  with no bundling of its own still pays for its parent's — so keep other
  threads, and root-scope operations above all, off the deadline-bearing
  subtree.
* **The contended remainder is the snapshot assembling a consistent view
  under fire** — bundle rebuilds at ~2 µs a pass while peers dirty the
  subtree — and what bounds it in practice is the privilege escalation,
  whose engagement had to be won by measurement.  Its
  `tags_owned == tags_total` gate was starved under pure age order:
  HIGHEST commits fastest, so its stamp is always the youngest and its tags
  were the first overwritten (8.5 of 11.8 probe ticks per slow commit lost
  to exactly that, and no reachable bound on the rebuild count).
  **Rule 0c** — a lower-priority tagger never overwrites a validated
  HIGHEST tag — removed the starvation at its source: organic grants rose
  ~30×, slow commits fell **62 → 15 per 900 s** (5.4σ), the MAX band moved
  24.3–34.6 → **20.4–23.7 µs**, p99 1.28 → 1.02 µs, and throughput gained
  4–6 % (the tag slots go quiet) with no tier paying for it — the shield
  needs a HIGHEST-owned slot, so peer-vs-peer tagging never reaches it.
  (Those are that A/B's own before/after, not the current state; for what
  the library measures today see the table above, where the same p99 reads
  896 ns.)  Triggering privilege *earlier* than the tag is a dead end,
  measured (null: grants neither spread nor stick while tags are being
  overwritten) and then subsumed — a HIGHEST tag now *is* the Reserved
  claim, so there is no earlier moment left; the knob and its OS-scheduler
  probe are gone.  The result is
  still an *observed* maximum, not a WCET — the gate's residual misses are
  side-word validation races, 2–5 ticks per slow commit.  (Whether folding
  the holder's priority class into the stamp's own PRIO field removes them is
  unresolved: the A/B that said so turned out to track run ORDER rather than
  the binary — see `design/RT_READINESS.md`.)  The record path
  makes no syscalls; the how and the dead ends are in
  [`tests/transaction_priority_mixed_test.cpp`](tests/transaction_priority_mixed_test.cpp),
  the lab notebook behind this section.

### The configuration

1. **Isolate the deadline-bearing thread** (`isolcpus`), everything else
   together on the housekeeping cores.
2. **`SCHED_FIFO` only on top of (1)** — alone it preempts the very peers it
   then waits behind, a measured priority inversion.
3. **A wait budget sized to the deadline**, kept well above the 300 µs
   reserve.
4. **`kame_pool_prewarm()` from that thread** before the time-critical
   section — the unwarmed first commit measured ~400 µs.
5. **`kame_pool_set_realtime_thread(KAME_RT_STRICT)` on that thread** — a
   commit frees cross-thread whenever a peer allocated on its subtree, and
   an ungated free batches ~1000 deep with one unlucky free paying the whole
   flush (3.5× of the slow-commit population until set).  KAME itself does
   not yet mark the thread; see [`kamepoolalloc`](../kamepoolalloc)'s
   contract.

### The one wait the budget cannot clip

The wait behind a **live privileged peer** is exempt by design — privilege is
the completion guarantee (it never expires above the LOW band), so waiting
the holder out is correctness.  Its bound is the holder's scheduling delay,
which is what configuration (1) is for: unpinned, MAX sticks at 12–13 ms
whatever the budget; pinned, the exemption never shows above the budget.

### HIGHEST, and its precondition

Everything above the first table row runs at NORMAL.  `Priority::HIGHEST`
additionally never parks and is immune to fair-mode, so each of its commits
landing inside a privileged peer's closure re-runs that closure.  Its
ceiling therefore has a precondition: **HIGHEST commit rate × longest peer
closure ≪ 1** — negligible at µs closures, divergent past the meeting point
(22 ms closures against a flat-out churner: 1.1 → 15.5 re-runs per commit).
KAME runs acquisition at NORMAL + OS elevation because its analysis closures
are ms-scale; use HIGHEST only where the precondition is a design property.

### No lock on the negotiation route

A losing transaction parks on `XWaitCell` (`xwaitcell.h`), and on macOS and
Linux the park is **mutex-less** — `__ulock_wait` / `futex(FUTEX_WAIT_PRIVATE)`
on a generation word.  The portable fallback's `std::mutex` has no priority
inheritance, so a high-priority committer could wait behind a preempted
low-priority one, unbounded once a medium thread interposes; removing the
mutex removes the question.  Throughput is measurably identical (the sleep
path is reached in 0.0001–0.05 % of commits).  Force the fallback with
`-DKAME_XWAITCELL_ULOCK=0` / `-DKAME_XWAITCELL_FUTEX=0`; `xwaitcell_test`
covers all three backends.

### Note: retry counts vs time

The model checking proves **starvation-freedom**, not a retry-count bound —
the right way around for a user, since the budget bounds *time* directly and
measured attempts sit at 1.002 mean / 5 worst.  A finite bound exists for
each checked configuration; none is established for a deployment, whose
arrival stream never drains.  "Retries are not bounded" means that, not
"retries can diverge": a losing transaction escalates to a privileged stamp
every peer must yield to.  That escalation is the *divergence* argument, not
the common case — it is probe-gated and fires a few times per million
commits, because a short CAS race resolves first.  Details in
[tests/VERIFICATION.md](tests/VERIFICATION.md).

## Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC.
The [paper](https://arxiv.org/abs/2608.12024) presents this verification in
full, including the spec-to-C++ fidelity argument; this section is the
repository-level map of the specs themselves:

- **Layer 1 — `atomic_shared_ptr`:** tagged-pointer CAS protocol with local/global reference counting, drain release, and `scoped_atomic_view` ([spec](tests/tlaplus/atomic_shared_ptr.tla)). Safety only — the bare primitive is intentionally *not* livelock-free.
- **Layer 2 — bundle/unbundle + commit:** 2-/3-level subtree bundling with a livelock-free privileged-TID negotiate mechanism, static and dynamic (online insert/release) ([2-level](tests/tlaplus/BundleUnbundle_2level_LLfree.tla), [3-level](tests/tlaplus/BundleUnbundle_3level_LLfree.tla), [dynamic](tests/tlaplus/BundleUnbundle_2level_LLfree_dynamic.tla)). Exhaustively model-checked **safe + livelock-free** without `CONSTRAINT` (the LL-free design makes the state space naturally finite — no artificial bound); the largest single exhaustive run reaches **~641 M distinct states** (3-level all-root, 15 h on the ISSP ohtaka supercomputer), over a billion across the LL-free configurations combined. (Raw state counts are **spec-version-specific** — they shift as the spec evolves, so cross-version comparison isn't meaningful; see [tests/VERIFICATION.md](tests/VERIFICATION.md) §3–§4 for the current-spec figures.) These are exhaustive results for the checked configurations (fixed thread counts and tree shapes), not an unbounded ∀-thread proof. The property is `<>AllDone` over a *draining* workload, so it is starvation-freedom, **not** a bound on retry counts — [Realtime behaviour](#note-retry-counts-vs-time) sets the two side by side.
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

## Citing

If this library or its verification artifacts contribute to your work,
please cite the paper:

```bibtex
@misc{kitagawa2026kamestm,
  title         = {Formally Verified Lock-Free Software Transactional Memory
                   for Scientific Measurement},
  author        = {Kitagawa, Kentaro},
  year          = {2026},
  eprint        = {2608.12024},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.other},
  doi           = {10.48550/arXiv.2608.12024},
  url           = {https://arxiv.org/abs/2608.12024}
}
```

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
