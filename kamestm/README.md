# kamestm

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20(MinGW%20%2B%20MSVC)-lightgrey)]()

Lock-free software transactional memory (STM) primitives — the
snapshot/transaction core from the [KAME](https://github.com/northriv/KAME)
measurement framework, extracted as a stand-alone, **header-only**
library plus one small `.cpp`.

Dual-licensed under your choice of **Apache 2.0 OR GPL-2.0-or-later**
so it embeds cleanly into permissive / proprietary projects (Apache
path) or links into GPLv2-only projects such as KAME itself (GPL path).

## What's in here

The library covers Layer 0 (atomic primitives) and Layer 1 (snapshot
+ transaction commit) of the KAME STM design:

| Header | Role |
|---|---|
| `atomic.h`, `atomic_prv_basic.h`, `atomic_prv_std.h` | Portable barriers + CAS over `std::atomic_thread_fence` |
| `atomic_smart_ptr.h` | Tagged-pointer lock-free `local_shared_ptr<T>` (the engine under every Snapshot) |
| `atomic_queue.h` | Lock-free MPMC queue |
| `mutex.h` | Lightweight wrappers around `std::mutex` / `std::shared_mutex` |
| `threadlocal.h` + `threadlocal.cpp` | `XThreadLocal<T, Tag>` with deterministic per-thread teardown |
| `xtime.h` + `xtime.cpp` | Monotonic time helpers used by Lamport-clock serial numbers |
| `transaction.h`, `transaction_definitions.h`, `transaction_impl.h`, `transaction_signal.h` | The STM core: `Snapshot<XN>`, `Transaction<XN>`, `Node<XN>`, `Talker<...>` |
| `transaction_negotiation.h`, `transaction_neg_impl.h` | Negotiated retries (`iterate_commit_negotiated`) |

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

The O(1) snapshot reads and CAS-based commits above require a shared pointer that is itself lock-free. `atomic_shared_ptr` (in `atomic_smart_ptr.h`, introduced in January 2006 as part of the 2.0-beta3 rewrite) provides this. It is a custom implementation of what C++20 calls `std::atomic<shared_ptr>`.

The core technique embeds a small **local reference counter** in the low bits of the pointer to the reference-control block — bits guaranteed zero by allocator alignment. `acquire_tag_ref_()` atomically increments this local counter via CAS to "pin" the pointer for reading; `release_tag_ref_()` decrements it. Between these two calls, even if another thread swaps the pointer, the object cannot be freed because the local count is non-zero. A separate **global reference counter** in the control block tracks long-lived ownership (copies held across scopes). Setters transfer any outstanding local count to the global counter before swapping, so `release_tag_ref_()` can fall back to decrementing the global counter if the pointer changed.

For types that inherit `atomic_countable` (notably `Payload`), the global reference counter is stored inside the object itself (**intrusive counting**), eliminating a separate heap allocation per shared-pointer instance. Non-intrusive types get an external control block (`atomic_shared_ptr_gref_`).

**Comparison with standard-library implementations (as of late 2024):**

| Implementation | Technique | Lock-free? |
|---|---|---|
| libstdc++ (GCC) | Spinlock on internal table | No — vulnerable to priority inversion |
| MSVC | Lock bit + `WaitOnAddress` | No — blocking under contention |
| libc++ (Clang) | Not yet implemented | N/A |
| KAME (2006–) | Tagged-pointer CAS | Yes — lock-free reads and writes |

On modern compilers (GCC 5.1+, Clang, MSVC), the CAS primitives delegate to `std::atomic` (`atomic_prv_std.h`). Hand-written assembly fallbacks for x86, PowerPC, and ARM remain in the tree for older toolchains.

**Multi-node consistency** is achieved through a *bundling* protocol: a parent packet absorbs child packets via multi-phase CAS protocol, making the entire subtree consistent under a single atomic pointer. A `m_missing` flag marks packets with stale children, driving re-bundling on demand.

**Collision backoff:** `Linkage::negotiate()` uses a `m_transaction_started_time` timestamp to impose a proportional wait on detected collisions, preventing live-lock under high write contention.

`iterate_commit_while(lambda)` lets the caller abort the retry loop (return `false` from the lambda to stop), enabling conditional transactions.

> **Caution:** Taking a nested `Snapshot` inside a transaction can trigger bundling, which may cause the transaction's CAS to always fail. This is not a data corruption issue but a liveness issue — the transaction retries indefinitely. This occurs when the `Snapshot` target is an ancestor of the transaction target, or when hard links exist (a child with two parents) and a `Snapshot` on one parent's tree interferes with the other. Use `tr[*node]` instead of a nested `Snapshot` in these situations.
>
> The hard-link case is formally modelled in `tests/tlaplus/BundleUnbundle_hardlink_*.tla` (sibling-parents and root-with-intermediate self-collision); see `tests/VERIFICATION.md` §5.

## Comparison with other STM designs

*The following comparison was written by Claude (Anthropic) based on analysis of the source code.*

Most widely-used STMs (GHC/Haskell `TVar`, Clojure `Ref`/`dosync`, ScalaSTM) are **flat**: the unit of transaction is a set of independent transactional variables. KAME's STM is instead **tree-structured** — the entire instrument node tree is the shared state, and snapshots are always subtree-consistent. This difference drives several design choices:

| Aspect | Flat STMs (Haskell, Clojure, ScalaSTM) | KAME STM |
|---|---|---|
| Conflict granularity | Per-variable | Per-packet (subtree root) |
| Read model | `readTVar` / `deref` inside transaction | `Snapshot` (outside) or `tr[*node]` (inside) |
| Consistency scope | Variables listed explicitly | Entire subtree, guaranteed by bundling |
| Commit log | Redo log or write set | Copy-on-write + CAS on single `Linkage` |
| Retry primitive | `retry` / `orElse` (Haskell) | `iterate_commit` / `iterate_commit_while` |
| Blocking | `retry` suspends on read-set change | No blocking; backoff via timestamp |
| Memory management | GC | Lock-free `atomic_shared_ptr` (ref-counted) |
| Hard real-time suitability | Limited (GC pauses) | Good (no GC, bounded CAS retries) |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to software backoff rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** These use a global version clock and per-object version stamps with a full read/write log per transaction. KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

## Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC:

- **`atomic_shared_ptr`:** tagged-pointer CAS protocol with local/global reference counting ([spec](tests/tlaplus/atomic_shared_ptr.tla))
- **`BundleUnbundle`:** subtree bundling/unbundling with modular serial arithmetic ([spec](tests/tlaplus/BundleUnbundle.tla))

Slide decks: [Layer 1 — atomic_shared_ptr](https://northriv.github.io/KAME/tests/tlaplus/doc/slides_layer1_en.html) ([JA](https://northriv.github.io/KAME/tests/tlaplus/doc_ja/slides_layer1.html)), [Layer 2 — Bundle/Unbundle + Commit](https://northriv.github.io/KAME/tests/tlaplus/doc/slides_layer2_en.html) ([JA](https://northriv.github.io/KAME/tests/tlaplus/doc_ja/slides_layer2.html))

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
  Snapshot allocation.  The STM core includes
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

Compile `kamestm/threadlocal.cpp` + `kamestm/xtime.cpp` into your
target.

A stand-alone `kamestm.pro` / `CMakeLists.txt` producing a
`libkamestm.dylib` is on the roadmap.

## Tests

Built by the `tests/` CMake scaffold and run with `ctest`
(`cmake -S tests -B build && cmake --build build && ctest --test-dir build`).
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
| `transaction_negotiation_test` | transactions of *different periodicities* — the slow loop never commits unless the fast loop yields a proportional backoff (`negotiate()`) |

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

Copyright (C) 2002-2026 Kentaro Kitagawa &lt;kitag@issp.u-tokyo.ac.jp&gt;,
The University of Tokyo, ISSP.

Both license grants explicitly require preservation of the copyright
notice and the choice-of-license clause when redistributing this
software, in source or binary form.
