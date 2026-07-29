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
every release from that year onwards.  Builds and passes all 11
standalone tests on macOS clang, Linux gcc/clang (64-bit + 32-bit),
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
Non-privileged contenders **park** (adaptive backoff / condition-variable
wait) instead of spinning, so the oldest / highest-priority transaction
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
| Hard real-time suitability | Limited (GC pauses) | Better (no GC pauses); livelock-free negotiation keeps the oldest Tx progressing — though CAS retry *counts* are not hard-bounded, so not hard-RT in a strict WCET sense |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to age-ordered privileged-Tx negotiation (the colliding losers yield to the oldest transaction) rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** Both use a global version clock and keep a read/write log per transaction, but differ on per-object metadata — TinySTM uses per-object version locks, whereas NOrec deliberately keeps *none* (it validates the read set by value against the global clock; the name is "No Ownership Records"). KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

## Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC:

- **Layer 1 — `atomic_shared_ptr`:** tagged-pointer CAS protocol with local/global reference counting, drain release, and `scoped_atomic_view` ([spec](tests/tlaplus/atomic_shared_ptr.tla)). Safety only — the bare primitive is intentionally *not* livelock-free.
- **Layer 2 — bundle/unbundle + commit:** 2-/3-level subtree bundling with a livelock-free privileged-TID negotiate mechanism, static and dynamic (online insert/release) ([2-level](tests/tlaplus/BundleUnbundle_2level_LLfree.tla), [3-level](tests/tlaplus/BundleUnbundle_3level_LLfree.tla), [dynamic](tests/tlaplus/BundleUnbundle_2level_LLfree_dynamic.tla)). Exhaustively model-checked **safe + livelock-free** without `CONSTRAINT` (the LL-free design makes the state space naturally finite — no artificial bound); the largest single exhaustive run reaches **~641 M distinct states** (3-level all-root, 15 h on the ISSP ohtaka supercomputer), over a billion across the LL-free configurations combined. (Raw state counts are **spec-version-specific** — they shift as the spec evolves, so cross-version comparison isn't meaningful; see [tests/VERIFICATION.md](tests/VERIFICATION.md) §3–§4 for the current-spec figures.) These are exhaustive results for the checked configurations (fixed thread counts and tree shapes), not an unbounded ∀-thread proof.
- **Hard-link topologies:** multi-parent / one-child races that reproduce and fix a production abort via a Phase-4 reachability gate and a Phase-3 skip-Null fix (`tests/tlaplus/BundleUnbundle_hardlink_*.tla`).

**Slide decks** — start at the **coverage overview** ([EN](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_overview_en.html) · [JA](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc_ja/slides_overview.html)), a hub linking every layer with a full coverage matrix. Individual decks (each with a Japanese counterpart under `doc_ja/`): [Layer 1](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer1_en.html), [Layer 2 base](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_en.html), [Layer 2 LLfree](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree.html), [3-level](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree_3level_en.html), [dynamic](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_layer2_LLfree_dynamic_en.html), [hard-link](https://northriv.github.io/KAME/kamestm/tests/tlaplus/doc/slides_hardlink_en.html).

C11 translations of each layer are verified with [GenMC](https://github.com/MPI-SWS/genmc) under the RC11 memory model: TLA+-derived tests (`tests/tlaplus/test_*.c`) and C++-derived protocol tests (`tests/cds_atomic_shared_ptr/`).

## Dependencies

- C++17 toolchain — gcc 9+, clang 10+, **and MSVC (cl)**.  All 11
  standalone tests build and pass on macOS clang, Linux gcc/clang
  (64-bit + 32-bit), Windows MinGW64 + lld, and Windows MSVC (cl 19.51).
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
(`cmake -S tests -B build && cmake --build build && ctest --test-dir build`).
(The two `*_mixed` throughput drivers are built but not `ctest`-registered —
they take command-line arguments and are run manually.)
Four layers, from primitive to whole-protocol:

**Atomic primitives** — exercise the lock-free building blocks directly:

| test | covers |
|---|---|
| `atomic_shared_ptr_test` | tagged-pointer `local_shared_ptr` load / store / CAS / swap under contention |
| `atomic_scoped_ptr_test` | single-owner scoped pointer + `local_weak_ptr` promotion |
| `atomic_queue_test` | lock-free MPMC queue |
| `mutex_test` | the `std::mutex` / `shared_mutex` wrappers |

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
