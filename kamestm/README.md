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
| Cost of an atomic multi-object commit | Per-variable logging, paid at commit | **p50 ≈ 439 ns + 94.5 ns × nodes**, measured 1–17 nodes ([below](#what-composability-costs)). In the *tail* it is free: with no peer on the subtree a five-node commit's MAX is 4.16 µs against a single-node commit's 4.94 µs |
| Hard real-time suitability | Limited (GC pauses) | No GC pauses, and a declared wait budget converts the tail into a chosen number — **measured** on a `PREEMPT_RT` host, MAX − budget = 3–7 µs at the shipped 20 ms budget (the host's own floor being 219 ns, measured), 38.3 M commits with zero over 3 ms at a 1 ms budget ([below](#realtime-behaviour)). Still not hard-RT in a strict WCET sense: CAS retry *counts* are not bounded, and the budget cannot bound the wait behind a live privilege holder — that one is the deployment's to bound, with core isolation |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to age-ordered privileged-Tx negotiation (the colliding losers yield to the oldest transaction) rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** Both use a global version clock and keep a read/write log per transaction, but differ on per-object metadata — TinySTM uses per-object version locks, whereas NOrec deliberately keeps *none* (it validates the read set by value against the global clock; the name is "No Ownership Records"). KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

## Realtime behaviour

**A commit's worst-case time is a number you choose.**  Declare a wait budget
(`ScopedWaitBudget`; KAME sets it from `XPrimaryDriver::downstreamWaitBudgetUS()`,
default 20 ms) and every wait inside the commit is clipped to it.  Measured
under contention on a `PREEMPT_RT` host — i5-7500, isolated core; workload
`transaction_priority_mixed_test`, the record commit against NORMAL / UI /
SCRIPTING peers — MAX − budget:

| budget | MAX − budget | acquisition | UI | SCRIPTING |
|---|---|---|---|---|
| 20 ms | **7.1 µs** | +2 % | −2 % | +2 % |
| 1 ms | 34 µs | +10 % | −13 % | −15 % |
| 200 µs | 19 µs | +1 % | **−94 %** | **−98 %** |

In the shipping configuration (`SCHED_FIFO` + isolation + PM-QoS, 20 ms
budget) MAX − budget is **3.0 µs**: the STM's contribution to the tail is a
few microseconds on top of whatever the host allows.

**And what the host allows is a configuration choice, not a property.**  Run
`latency_floor` — the same clock and histogram with no STM in the loop — before
quoting any absolute number here.  On this host it moved by three orders of
magnitude across three steps: as found (the kernel command line had silently
lost its isolation across a reboot) MAX 67.9 µs; with `isolcpus`/`nohz_full`
actually taking effect, 17.0 µs; and under `tests/with_pmqos`, which holds
`/dev/cpu_dma_latency` at 0 for the child's lifetime, **219 ns — with not one
sample over a microsecond in 370 million.**  The middle rung is the package
leaving a deep C-state, invisible to every counter the kernel keeps.  `rtla
osnoise` reports 17 µs Max Single here, i.e. exactly that rung and nothing
about the 219 ns underneath it.

**Three rungs, not one number — subtract the one whose configuration matches
your run.**  This is easy to get wrong and was got wrong here: a commit MAX
measured *with* isolation was once compared against the *un-isolated* rung and
declared level with the machine.  Lined up correctly, the same measurements say
the opposite:

| configuration | machine floor | commit MAX | difference |
|---|---|---|---|
| isolated, no PM-QoS | 17.0 µs | 66.7 µs | **49.7 µs** |
| isolated + PM-QoS | 219 ns | 53.1 µs | **52.9 µs** |
| + FIFO and per-thread pinning, `invol` = 0 | 219 ns | 40.8 µs | **40.6 µs** |

The floors span **77×** and the differences span 1.3×.  That near-invariance is
what identifies it: **40–50 µs of the tail is the STM's own**, and the sections
below say which part — the retry path, three quarters of the worst commit in
the last row, measured directly.  The check is general and cheap: if a tail is
really the machine, deleting the machine deletes the tail.  Here deleting all
but 219 ns of it, and then every involuntary context switch as well, left
40.6 µs standing.

Two things earned those numbers.  First, instrumentation: the previous
constant (~200 µs of overshoot at every budget) was **the timed wait's
wake-up cost, not the STM** — the worst commit was one `cell.wait()` asked
for 198 µs that returned 696 µs later with 6 µs of STM work in the whole
commit, and the scheduling class dominates the cost (5.3×) with C-state
exit buying its 6× only on top of it.  Second, the consequence: a timed
wait must never be armed to land *on* the deadline, so every budgeted sleep
now stops `KAME_NEG_SPIN_TAIL_US` (300 µs) short and the remainder is
polled — which also observes the blocker clearing immediately.  Unbudgeted
callers run the pre-existing code unchanged (measured −0.8 %, noise).

**Keep the budget well above the 300 µs reserve** — the percentages in the
table's last row are the cliff: a budget at or below the reserve never
sleeps, never backs off the linkage, and starves the deferrable tiers.  At
the shipped 20 ms the reserve is 1.5 % of the budget and free.  Throughput
otherwise *rises* as the budget falls (a clipped commit stops sleeping and
retries; the clip rate stays ~0.32 %).  At length, 300 s at a 1 ms budget
(measured before the reserve, so with the old +216 µs tail): **38.3 M
commits, MAX 1.288 ms, zero over a 3 ms deadline**, every other role
healthy.  Attempts per commit: mean 1.002, worst 5.

### What composability costs

A commit here is *composable*: it is atomic over a whole subtree, and every
reader of that subtree sees it whole or not at all.  That is the expensive part
of the design, so it is worth knowing the price rather than assuming it.
`KAME_MIX_LEAVES=0` gives the control — every subtree becomes a bare node, a
`Transaction` on it is a `SingleTransaction`, and `bundle()` is never called
(the harness prints the pass count, and it is 0.00).  Same thread, same
isolated core, same peers, 60 s each:

| nodes in the commit | p50 | p99 | p99.9 | **MAX** | commits/s |
|---|---|---|---|---|---|
| 1 (no bundling at all) | 448 ns | 448 ns | 768 ns | **31.3 µs** | 323 k |
| 2 | 640 ns | 640 ns | 1.28 µs | **34.1 µs** | 148 k |
| 5 | 896 ns | 1.28 µs | 3.07 µs | **40.6 µs** | 119 k |
| 17 | 2.05 µs | 2.56 µs | 12.3 µs | **50.0 µs** | 49 k |

The median is linear and the fit is boring in the good way:
**p50 ≈ 439 ns + 94.5 ns × nodes**, within 2 % at every point, with the
intercept landing on the 448 ns single-node cost.  So a node costs ~94 ns to
carry inside an atomic commit, and a **17-node atomic commit closes in
2.05 µs**.

The line that matters for realtime is the last column, and it barely moves:
**17× the nodes buys 1.6× the worst case.**  Whatever sets the tail, it is not
the topology.

### The tail is contention, and contention is yours to remove

`KAME_MIX_DISJOINT=1` is the control that settles it, and it is a *topology*
control, not a load one: every peer keeps running on the same core at the same
rate, allocating and freeing exactly as much, and only what they *touch*
changes — cross-thread writes move off the acquiring subtree, and the two
root-scope operations drop to a sibling scope, since a root scope bundles the
acquiring subtree whatever it writes.  Turning the peers off instead would
delete the machine load along with the conflict and prove nothing.

| nodes | peers touch the subtree? | attempts | bundle | unbundle | p99.999 | **MAX** | commits/s |
|---|---|---|---|---|---|---|---|
| 5 | yes | 3 | 0.35 | 1.50 | 28.7 µs | **48.0 µs** | 121 k |
| 5 | **no** | **1** | 0.00 | 0.00 | 4.10 µs | **4.16 µs** | 302 k |
| 1 | yes | 3 | 0.00 | 1.88 | 20.5 µs | **32.3 µs** | 313 k |
| 1 | **no** | **1** | 0.00 | 0.00 | 3.07 µs | **4.94 µs** | 952 k |

**Removing the conflict collapses the worst case 11.5×** (6.5× at one node),
and takes it to **zero commits over 10 µs in 18 M and 57 M respectively**.
Throughput rises 2.5–3.0× at the same time, so the conflict was certainly
real — it simply *was* the tail.

Two readings fall out.  **Composability contributes nothing to the tail**: with
no conflict, the five-node commit's MAX (4.16 µs) is *lower* than the
single-node commit's (4.94 µs).  Bundling five nodes is free in the tail and
costs 94 ns per node in the median.  And a single-node commit is not
conflict-free by construction — its 32.3 µs comes with `unbundle()` running
1.88 times per slow commit, because `devA` is a child of the root and a peer's
root-scope `Snapshot` bundles it whether it has children of its own or not.
**A subtree with no bundling of its own still pays for its parent's.**

So the deployment lever is topological, and it is a large one: keep other
threads off the deadline-bearing subtree, and in particular keep *root-scope*
snapshots and transactions away from it, since those bundle everything beneath
them.  What remains at zero conflict is ~4–5 µs — 11× the 448 ns median and 23×
the 219 ns floor, with the negotiator, the scheduler and the machine all
already excluded.  That residue is not yet attributed; the allocator on the
copy-on-write path is the standing candidate.

### The one wait the budget cannot clip

The exception is the wait behind a **live privileged peer** — privilege is
the completion guarantee (it never expires above the LOW band of LOWEST /
UI_DEFERRABLE / SCRIPTING), so waiting a holder out is correctness.  (It is
also rarer than it looks: instrumentation found **zero** exempt rounds
across 17,274 slow commits of the pinned workload above — the overshoot
those commits carried was the wake-up cost, not this wait.)  Its two
terms are the deployment's to bound:

* **The holder's scheduling delay** — bound it with isolation: every other
  STM thread together on the housekeeping cores.  Unpinned, MAX sticks at
  12–13 ms whatever the budget; pinned, the residue vanishes (the table
  above).  `SCHED_FIFO` helps only on top of isolation — alone it preempts
  the very holders it then waits behind, a measured priority inversion:
  contenders collapsed to ~150 commits/s and the elevated thread still took
  50.9 ms on its own worst commit.
* **The holder's closure re-runs under HIGHEST churn** — the precondition
  below.  Not a scheduling problem; isolation does not touch it.

### The configuration

1. **Isolate the deadline-bearing thread** (`isolcpus`), everything else
   together on the housekeeping cores.
2. **`SCHED_FIFO` only on top of (1).**
3. **A wait budget sized to the deadline**, and kept well above the 300 µs
   reserve — MAX = budget + ~7 µs, or ~3 µs with (1), (2) and PM-QoS.
4. **`kame_pool_prewarm()` from that thread** before the time-critical
   section: a commit clones a payload, so the allocator is on the deadline
   path, and the unwarmed first commit measured ~400 µs.

### HIGHEST, and its precondition

Everything above runs at NORMAL.  `Priority::HIGHEST` additionally never
parks at all; same host and roles, 120 s:

Isolated core with the tick verified stopped (`LOC` = 0 over the window),
`SCHED_FIFO`, per-thread pinning, `with_pmqos`, and the pool's realtime
contract honoured — against the 219 ns floor above.  120 s each:

| tier | p50 | p99 | p99.9 | p99.999 | **MAX** |
|---|---|---|---|---|---|
| HIGHEST (the library's ceiling) | 896 ns | 1.02 µs | 3.07 µs | 24.6 µs | **40.8 µs** |
| NORMAL, 20 ms budget | 768 ns | 1.02 µs | 7.34 ms | 20.97 ms | **20.03 ms** |

The HIGHEST row is 14.3 M commits with **zero involuntary context switches**,
and **nothing over 50 µs at all**.  Note it is a *contended* figure — the peers
in this workload deliberately write into the acquiring driver's own subtree,
and taking them off it drops the same MAX to 4.16 µs
([below](#the-tail-is-contention-and-contention-is-yours-to-remove)).  The NORMAL row is the budget doing its job:
0.054 % of commits reach it, and MAX *is* it.

Read the two rows as one regime only for the scheduling and the host; the
NORMAL row predates the verified-shape harness below and has not been re-taken
in it.  An earlier HIGHEST row published here — p99 1.79 µs, p99.9 20.5 µs, MAX
53.1 µs — has been **replaced rather than kept beside this one**: it was taken
before the harness could report its own OS arm, so its scheduling is not
established, and every column of it is worse.  p99.9 in particular is 6.7×
better here, which is the size of effect that says the two are different
configurations and not two samples of one.

Getting there took two fixes that are worth stating because neither is
obvious.  **The allocator owned 3.5× of the slow-commit population**: a commit
frees *cross-thread* whenever a peer allocated on its subtree, and an ungated
cross-thread free batches to `CAP=1024` with one unlucky free paying the whole
flush.  `kame_pool_set_realtime_thread(KAME_RT_STRICT)` on that thread takes
commits over 50 µs from 28.8 to 8.3 per million and MAX from 92.6 to 66.7 µs
(both arms isolated, no PM-QoS, so both against the 17.0 µs floor — 66.7 µs is
**3.9× the machine**, not level with it); process-wide realtime mode (31.0) and
`KAME_RT_DEFER` (29.3) buy nothing, only STRICT.  The tests default to it.  **KAME
does not yet mark that thread** — `kame/main.cpp` sets the process-wide mode
only — so it is an outstanding precondition, not a shipped property; see
[`kamepoolalloc`](../kamepoolalloc)'s contract.  And **the host had to be made
quiet first**, which is the ladder above.

p99.9 was for a long time the number that *would* not move: it stayed in the
[16.4, 20.5) µs bucket through core isolation, the tick stopping, C-states off
and the allocator fix alike, which is what argued it was the STM's own and not
the host's.  **One thing did move it — per-thread pinning, 20.5 → 3.07 µs**,
and that is a scheduling fix, not an STM one.  The conclusion survives with a
smaller number: at 3.07 µs it is still **14× the 219 ns floor**, and the tail
above it still is not the machine.

What it is not is settled either way.  Instrumenting the negotiator: over 8,007
slow commits and in the worst one individually, `sleeps`, `spins`, `slept_ns`,
exempt rounds and wait overshoot are **all zero**, and 51,796 ns of a 51,796 ns
commit is unaccounted for by the negotiation machinery.  It is not waiting for
anyone.

**It is the retry path, measured.**  Timing `iterate_commit`'s phases
separately (`KAME_MIX_PHASE=1`), in the verified shape above over 1,725 slow
commits: snapshot 4,769 ns, payload write 1,080 ns, the *successful* commit
3,500 ns — and **failed attempts plus the re-snapshot they trigger,
9,220 ns**.  The worst commit splits **998 / 1,588 / 7,925 / 30,245** of
40,812 ns with **56 ns unattributed**: three quarters of it is failed attempts,
in a run with zero involuntary context switches.

So **a failing attempt costs ~11× a succeeding one** (9,220 ns over 0.56 failed
attempts against 3,500 ns for the one that worked), which is where the
arithmetic pointed and is now direct.

*What* it spends that on is a separate question, and the first answer was
wrong.  "A re-snapshot is multi-nodal, so it re-bundles the subtree and throws
the pass away" fits the shape — only a peer whose transaction *spans* the
acquiring subtree provokes it (5× the slow-commit rate), the cost is
path-shaped so 4× the leaves leaves the magnitude alone, and a root `Snapshot`
at 42 kHz, which bundles but does not span, provokes nothing.  It does not fit
the *magnitude*: two bundle+unbundle passes at the cost of a whole successful
commit come to 4.8 µs against 15.6 µs measured, short by 3.2×, and
`bundle_cas_retries` is 0.00 so it is not spinning either.

`bundle()` and `unbundle()` are now timed directly and differenced across the
same boundaries as the retry phase itself, which needs no arithmetic at all:
**re-bundling is 40 % of the retry cost on average and 24 % in the worst
commit, not the whole of it** — 30–40 % across every RT-host and container run
taken.  The remaining ~two thirds are unattributed, and the harness prints the
share so the next hypothesis has to clear the same bar.

(Pair the two terms only within one run, and only at matching scope: a
whole-commit bundle total also contains the snapshot's and the successful
commit's bundles — 45 % of all bundling here — and set against the retry phase
alone it reads above 100 %.)

Left as a characterisation rather than a fix: it is 20–50× inside a 1 ms
deadline.  Narrowing it further means separating the failed `commit()` from the
`++tr` inside it, which the public API fuses.

**Why the privileged escalation does not rescue it.**  The obvious objection is
that the STM has a mechanism for exactly this: a transaction that keeps losing
claims privilege and every peer defers to it.  The claim gate carries **no
priority term** — HIGHEST claims on the same terms as anyone — and it converts
every verdict it is given.  It is simply never given one here.  Counting the
gates (`KAME_STM_NEG_DIAG` `ll_*`) rather than reading them — four gates,
mutually exclusive and exhaustive, over two RT-host runs of 120 s:

| gate | share of the probe ticks it blocks |
|---|---|
| retry threshold `clamp(sig_C·2, 3, nproc)` | **57–63 %** |
| the per-linkage window reset | 26–34 % |
| `tags_owned == tags_total` | 9–11 % |
| `m_tagged_linkages.empty()` — outside the probe | 7–9 % (but ~37 % on a shared container, where a plain CAS loss never reaches a negotiation at all) |

Across 8.3 M and 15.9 M commits the probe ran 63,616 and 45,967 times and
reached a verdict **once, in one of the two runs**.

Why the threshold binds is not a fixed answer, and asking it twice gave two.
`my_tx_retries` at a probe tick peaks at **2** on a shared container, **3** on
the isolated pair, and **4** on the housekeeping pair, against a threshold of
**4** throughout.  So the workload sits *exactly on the boundary*, and which
side it lands is set by how disturbed the host is:

| run | peak retries | threshold | verdicts in ~14 M commits |
|---|---|---|---|
| container | 2 | 4 | 0 |
| isolated pair, `invol` = 0 | 3 | 4 | 0 |
| housekeeping pair, `invol` = 137 | 4 | 4 | 3 |

Privilege fires when the host disturbs the thread and not otherwise — defensible
as a design (the probe is for pathological cases) but it does mean **privilege
is not what bounds the clean-host tail**.  The mean `my_tx_retries` at a tick is
**0.12** in every one of these: the probe is nearly always looking at a *first
attempt*.  Every verdict ever reached, on any host, converted to a claim and a
grant — 4 for 4.

Do not shortcut this from the attempt count — `snapshot()`'s own retry loop
increments the same counter live and restores it on scope exit, so a tick from
inside it sees more retries than any attempt count predicts.  Nor from one
host: "peaks at 2" was a contention level published once as an invariant, and
the next run refuted it.  The harness prints the margin and says REACHABLE or
UNREACHABLE outright, per run, for that reason.

The shares themselves come from four runs whose MAXes span 41 µs to 4.0 ms and
which agree to a few points across all of them — which is what makes them a
property of the workload rather than of the schedule.  Two of those four were
SCHED_OTHER with all threads sharing two cores, because an outer `chrt -f 20`
did nothing (every thread body resets its own policy, deliberately, so the
elevation was demoted at thread start — the harness now adopts an inherited RT
priority and says so).  Their *latencies* are not quoted anywhere.

So this tail is a privilege **absence**, not a privilege failure: the probe is
calibrated for sustained mutual livelock, and a 3-attempt CAS race that
resolves on its own is not that.  The one verdict the whole run produced
converted to one claim and one grant.  Lowering the threshold is the lever if
one is ever wanted.

The ceiling's precondition: **HIGHEST commit rate × longest peer closure
≪ 1.**  HIGHEST is also immune to fair-mode, so each of its commits landing
inside a privileged peer's closure re-runs that closure — negligible at the
µs closures above, divergent past the meeting point (22 ms closures against a
flat-out churner: 1.1 → 15.5 re-runs per commit, the holder pinned by
arithmetic, everything behind it waiting exempt from any budget).  KAME runs
acquisition at NORMAL + OS elevation because its analysis closures are
ms-scale; use HIGHEST only where the precondition is a design property.

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
the measured attempts sit at 1.002 / 5.  A finite bound exists for each
checked configuration; none is established for a deployment (the checking is
per-configuration, and the specs drain a fixed workload while a deployment
faces a continuing arrival stream).  "Retries are not bounded" means that —
not "retries can diverge": a losing transaction escalates to a privileged
stamp all peers, first-attempt ones included, must yield to.  Read that as the
*divergence* argument it is, not as a description of the common case: the
escalation is **probe-gated**, and in the measured deployment mix it fires
about once per million commits, because a short CAS race resolves before the
probe's threshold is reached ([above](#realtime-behaviour)).  Retries stay
finite there by winning, not by escalating.  Details in
[tests/VERIFICATION.md](tests/VERIFICATION.md).

## Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC:

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
