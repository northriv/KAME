# Formal Verification of KAME's Lock-Free STM Primitives

## Overview

Three complementary verification approaches covering the full stack.
TLA+ layers: Layer 0 = `atomic_shared_ptr` protocol; Layer 2 = commit + 2/3-level bundle/unbundle.

| Layer | Tool | Target | What it verifies |
|---|---|---|---|
| Memory model | GenMC v0.16.1 (RC11) | `atomic_shared_ptr` | `memory_order_relaxed` / `acq_rel` safety under weak memory |
| Layer 1 (TLA+) | TLC | `atomic_shared_ptr` | Tagged-pointer refcounting protocol (scan/CAS/reset) with ABA + non-atomic reads |
| Layer 2 (TLA+) | TLC | 2-level bundle/unbundle | Multi-phase CAS + LL-free priority for 2-level subtree (Parent → Children)/ 3-level subtree (Grand → Parent → Children); recursive inner bundle; static + dynamic |
---

## 1. GenMC: `atomic_shared_ptr` Memory Model Verification

**Directory:** `tests/cds_atomic_shared_ptr/`

### What it tests

The core lock-free reference counting protocol from `kame/atomic_smart_ptr.h`, extracted into
standalone C11 programs with all `memory_order` annotations exactly matching the original:

| Operation | Ordering in original |
|---|---|
| `compare_set_weak` success | `memory_order_acq_rel` |
| `compare_set_weak` failure | `memory_order_relaxed` |
| `load_shared_` global refcnt `fetch_add` | `memory_order_relaxed` |
| `compareAndSwap_` transfer `fetch_add` | `memory_order_relaxed` |
| `compareAndSwap_` rollback `fetch_add(negative)` | `memory_order_relaxed` |
| `compareAndSwap_` final `fetch_sub` (step 6) | `memory_order_acq_rel` |
| `release_tag_ref_` fallback `decAndTest` | `memory_order_acq_rel` |
| `local_shared_ptr::reset` `decAndTest` | `memory_order_acq_rel` |

### Test files

| File | Scenario | Threads |
|---|---|---|
| `cds_test_load.c` | 3 threads concurrently `load_shared_()` + `release_tag_ref_()` | 3 |
| `cds_test_cas.c` | Reader (`load_shared_`) vs writer (`compareAndSwap_<false>`) | 2 |
| `cds_test_multi_cas.c` | 2 threads race `compareAndSet` on same target | 2 |

### Key modeled details

- **Tagged pointer scheme**: `CAPACITY=8`, pointer in upper bits, local refcount in lower 3 bits
- **Intrusive reference counting**: `Obj.refcnt` as `_Atomic(uintptr_t)`
- **`acquire_tag_ref`**: CAS loop incrementing local refcount in tagged pointer
- **`release_tag_ref`**: CAS to decrement local refcount; falls back to global `decAndTest(acq_rel)` if pointer was swapped
- **`load_shared_`**: acquire_tag_ref → `fetch_add(1, relaxed)` on global refcnt → release_tag_ref
- **`compareAndSwap_`**: pre-increment newr → acquire_tag_ref → transfer local→global → CAS → release old
- **`compare_and_set` (NOSWAP=true)**: callers pre-acquire global refs before spawning threads (matching real `const local_shared_ptr<T>&` calling convention)

### Assertions

- No use-after-free (`assert(p->destroyed == 0)` while holding reference)
- No double-free (refcnt reaches 0 exactly once)
- No memory leaks (all objects destroyed, all refcounts reach 0)
- GenMC additionally checks: no data races on non-atomic accesses

### Results

| Test | Complete executions | Blocked executions | Wall time | Result |
|---|---|---|---|---|
| 1: concurrent load | 5,757 | 6,428 | 0.5s | **Pass** |
| 2: load + CAS | 240 | 7 | 0.05s | **Pass** |
| 3: multi CAS | 464,164 | 705,296 | 145s | **Pass** |

"Blocked executions" are partial executions pruned by GenMC's DPOR algorithm as redundant.

### Key findings

1. **`fetch_add(1, relaxed)` in `load_shared_` is safe** — the preceding `acq_rel` CAS in `acquire_tag_ref_` pins the pointer before the global refcount increment.
2. **`relaxed` transfer/rollback in `compareAndSwap_` is safe** — the final `acq_rel` CAS that installs the new pointer synchronizes-with subsequent observers.
3. **The tagged-pointer local refcount scheme prevents premature deallocation** — no execution was found where a thread observes a freed object.

### Build & run

```bash
cd tests/cds_atomic_shared_ptr

# One-time: build GenMC (requires LLVM 15-20, NOT 22+)
brew install llvm@20 hwloc
git clone https://github.com/MPI-SWS/genmc.git
cd genmc && mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@20 \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/lib" \
  -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
cd ../..

# Run all tests
make run

# Or individually
genmc/build/genmc --rc11 --unroll=5 /full/path/to/cds_test_load.c
genmc/build/genmc --rc11 --unroll=5 /full/path/to/cds_test_cas.c
genmc/build/genmc --rc11 --unroll=5 /full/path/to/cds_test_multi_cas.c
```

**Note:** GenMC compiles source files directly via its LLVM frontend. Pass absolute paths.
The `--unroll=5` flag bounds CAS retry loops. Increase if GenMC warns about insufficient unrolling.

---

## 2. TLA+ Layer 1: `atomic_shared_ptr` Protocol Verification

**Directory:** `tests/tlaplus/`  
**Spec:** `atomic_shared_ptr.tla`

### What it tests

The tagged-pointer reference counting protocol from `kame/atomic_smart_ptr.h` under sequential
consistency. Complements GenMC (which checks memory ordering) by exhaustively exploring all
thread interleavings of the higher-level CAS protocol.

### Modeled operations

| Operation | C++ source | Key detail |
|---|---|---|
| `reserve_scan_()` | Lines 462-488 | CAS on full word (ptr + local_rc); **non-atomic read modeled as 2 steps** |
| `scan_()` | Lines 494-503 | reserve_scan_ + global_rc++ + leave_scan_ |
| `leave_scan_()` | Lines 512-531 | CAS to dec local_rc; **fallback to global_rc-- if ptr changed** |
| `compareAndSwap_()` | Lines 556-603 | 6 phases: pre-inc, reserve, check/mismatch, transfer, CAS, cleanup |
| `local_shared_ptr::reset()` | | global_rc-- with free on zero |
| ABA Recycle | (synthetic) | Freed objects reallocated at same address to test ABA |

### Key modeling decisions

1. **Non-atomic read splitting**: `pref_()` and `refcnt_()` are two separate `m_ref.load()` calls in C++. Modeled as `rs_read_ptr` → `rs_read_rc` (two interleaving steps). Another thread can change `m_ref` between them; the subsequent CAS catches the inconsistency.
2. **ABA recycling**: Freed objects can be "recycled" (same pointer value reappears). Verifies that the tagged-pointer local_rc prevents ABA — the CAS compares the full word including the refcount bits, not just the pointer.
3. **CAS mismatch = return false**: `compareAndSwap_` with `pref != oldr` returns false immediately (no retry). Only the *inner* CAS (on `m_ref`) retries on failure.

### Verified invariants (6)

| Invariant | Description |
|---|---|
| `MemorySafety` | After reserve_scan_ succeeds, object is not freed until leave_scan_ completes |
| `NoUseAfterFree` | Objects held by local_shared_ptrs are not freed |
| `GlobalRCNonNeg` | Global refcount >= 0 for live objects |
| `FreedImpliesZeroRC` | Freed objects have global_rc = 0 |
| `InstalledNotFreed` | Object currently in atomic_shared_ptr is not freed |
| `TypeOK` | Type invariant |

### Results

| Config | Threads | Objects | Features | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|---|---|
| step1_scan_only | 2 | 2 | non-atomic read | 14,280 | - | 2s | **Pass** |
| step2_scan_plus_cas | 2 | 2 | CAS + ABA + non-atomic read | 115,714,315 | 167 | 9min | **Pass (complete)** |
| step3_concurrent_cas | 3 | 2 | CAS + ABA + non-atomic read + symmetry | 71,335,517+ | - | 10min | **Pass (partial, no violations)** |

### Key findings

- **ABA does not cause memory safety violations**: The TLA+ model permits object recycling at the same pointer value (sound over-approximation). For `scan_()`/`load_shared_()`, where the thread holds no reference during `acquire_tag_ref`, ABA recycling can in principle occur; the model verifies that reference counting invariants are preserved even in this case. For `compareAndSwap_()`, the caller holds a `local_shared_ptr` to the old value, so ABA cannot occur in practice (the old object is never freed). 115.7M states verified with no memory safety violations.
- **Non-atomic read is safe**: Inconsistency between the two loads is always caught by the subsequent CAS (which compares the full atomic word).
- **`leave_scan_` fallback branch is essential**: When the pointer is swapped between reserve and leave, the `global_rc--` fallback (lines 526-529) correctly maintains refcount invariants. Without it, refcount leaks or use-after-free would occur.

---

## 3. TLA+ Layer 2, 2-level bundle/unbundle Verification:

**Directory:** `tests/tlaplus/`  
**Specs:** `BundleUnbundle_2level_LLfree.tla`, `BundleUnbundle_2level_LLfree_dynamic.tla`

### What it tests

The multi-phase CAS bundle/unbundle protocol for a 2-level tree (Parent → {Child1, Child2}),
plus the livelock-free (LL-free) priority mechanism. Models `bundle()`, `unbundle()`,
`commit()`, and `snapshot()` from `kame/transaction_impl.h` for the 2-level case.

### Priority (LL-free) mechanism

`priorityTag[n] ∈ {Null} ∪ ({0..MaxIter} × Threads)` per node. Older transaction (smaller
iter, then smaller tid) wins. CAS failure → set own tag if older (`TagAfterFail`). Older tag
blocks younger threads (`CanProceed`). `PreemptTag` lets an active older thread snatch a tag.
Tags cleared only on commit success (`ClearMyTags`). Mirrors `m_priority_tidstamp` /
`ScopedNegotiateLinkage` in `transaction_impl.h`.

### Specification generations

| Generation | Serial | Finiteness | Liveness | Status |
|---|---|---|---|---|
| Gen 1 (modular) | `% MaxSerial` wrap | Structural | Fails (SerialWrapAround violated) | Counter-example: shows LL-free necessary |
| Gen 2 (Nat+CONSTRAINT) | Nat monotone | `CONSTRAINT SerialBound` cutoff | Not proven | Shows Nat alone insufficient |
| **Gen 3 (LL-free)** | Nat monotone | **Structural** (priority bounds retries) | **Proven** | **Current reference** |

### Static spec: `BundleUnbundle_2level_LLfree.tla`

- 4-phase bundle: collect sub-packets → CAS parent (missing=TRUE) → CAS each child to
  BundledRef → CAS parent (missing=FALSE, finalize)
- Unbundle for commit: mark child slot Null (missing=TRUE) → restore child priority
- 3 atomicity granularities: `coarse` / `fine` / `superfine` (most C++-faithful)

### Dynamic spec: `BundleUnbundle_2level_LLfree_dynamic.tla`

Extends the static spec with online child insertion (`insert(online=true)`) and release.
Thread roles configurable via `InsertThreads`, `RootThreads`, `LeafThreads`, `ReleaseThreads`.

### Verified invariants

| Invariant | Description |
|---|---|
| `SnapshotConsistency` | If node has `missing=FALSE`, all sub-packets exist |
| `NoPriorityLoss` | Non-root nodes always have `hasPriority=TRUE` or `bundledBy≠Null` |
| `BundleRefConsistency` | If child is bundled to parent, parent has priority |
| `MissingPropagation` | `missing=TRUE` propagates to all ancestors |
| `TerminalPayloadCheck` | At termination each child received exactly the expected increments |
| `EventuallyAllDone` (PROPERTY) | All threads eventually complete — formal liveness proof |

### Selected results (Gen 3, CONSTRAINT-free exhaustion)

| Config | Threads | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|
| 2-thread coarse | 2 | 665,218 | 89 | 28 s | **Pass + liveness** |
| 2-thread superfine | 2 | 2,676,196 | 129 | 3:12 | **Pass + liveness** |
| 3-thread superfine confC (all-root) | 3 | 137,333,348 | 96 | 6:35 | **Pass** (ohtaka) |
| MaxCommits=2 superfine | 2 | 127,586,599 | 311 | 4:40 | **Pass** (ohtaka) |
| dynamic release superfine live | 2 | 413,884,516 | 320 | 7:13 | **Pass + liveness** (ohtaka) |

Full results: `tests/tlaplus/doc/verification_log.md`

### Build & run

```bash
cd tests/tlaplus
# tla2tools.jar included; requires OpenJDK 21+

# 1-thread sanity (< 1 s)
java -XX:+UseParallelGC -Xmx8g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_2level_LLfree_coarse_mc.cfg \
  BundleUnbundle_2level_LLfree.tla

# 2-thread coarse with liveness (~30 s)
java -XX:+UseParallelGC -Xmx14g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_2level_LLfree_coarse_mc.cfg \
  BundleUnbundle_2level_LLfree.tla
```

---

## 4. TLA+ Layer 2, 3-level bundle/unbundle Verification:

**Directory:** `tests/tlaplus/`  
**Spec:** `BundleUnbundle_3level_LLfree.tla`

### What it tests

Same protocol as Layer 1 extended to a 3-level tree (Grand → Parent → {Child1, Child2}).
Additionally verifies recursive bundling (snapshot of Grand triggers inner bundle of Parent)
and multi-level unbundle walk.

### Additional coverage vs Layer 1

- **Recursive inner bundle**: `InnerPhase2/3/4` model the inner `bundle()` call when
  `snapshot(Grand)` encounters Parent with `missing=TRUE`
- **Multi-level unbundle walk**: `commit(Child)` when Child is bundled 2 levels deep
  (Child → Parent → Grand); walk traverses `bundledBy` chain up to Grand
- **`BundleChainValid`** / **`BundledByCorrect`** invariants: structural correctness of
  the 3-level `bundledBy` chain

### Verified invariants (adds to Layer 1)

| Invariant | Description |
|---|---|
| `BundleChainValid` | Bundled node's `bundledBy` target is priority or itself bundled |
| `BundledByCorrect` | `bundledBy` always points to the structural parent |
| `GrandAlwaysPriority` | Root (Grand) node always has priority |

### Selected results (Gen 3, CONSTRAINT-free exhaustion)

| Config | Threads | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|
| 2-thread coarse | 2 | 1,497,098 | 98 | 1:35 | **Pass + liveness** |
| 2-thread superfine | 2 | 14,109,731 | 148 | 19:13 | **Pass + liveness** |
| 3-thread superfine confC (all-root) | 3 | 640,894,951 | 88 | 15:25 | **Pass + liveness** (ohtaka) |

Full results: `tests/tlaplus/doc/verification_log.md`

### Build & run

```bash
cd tests/tlaplus

# 1-thread sanity (< 1 s)
java -XX:+UseParallelGC -Xmx8g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_3level_LLfree_1thr_superfine_mc.cfg \
  BundleUnbundle_3level_LLfree.tla

# 2-thread coarse with liveness (~2 min)
java -XX:+UseParallelGC -Xmx14g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_3level_LLfree_coarse_mc.cfg \
  BundleUnbundle_3level_LLfree.tla
```

Large configs (3-thread superfine) require a supercomputer; see
`tests/tlaplus/doc/ohtaka_handoff.md`.

### Liveness (both Layer 1 and Layer 2)

**Lock-free + livelock-free formally proven** via `EventuallyAllDone` PROPERTY. Priority
gating makes unbounded retries structurally impossible: serials increment monotonically, the
state graph is acyclic, and TLC terminates without `CONSTRAINT`. See
`tests/tlaplus/doc/proof_semantics.md` §2–§4 for the full argument.

Wait-free is not claimed — CAS-retry-based with fairness (`WF_vars`) required for per-thread
liveness. See `proof_semantics.md` §7.

---

## 5. Runtime Stress Tests (C++)

**Directory:** `tests/`

Complementing the exhaustive model checking above, the C++ stress tests exercise the real
implementation under high contention with many iterations, catching bugs that depend on
timing, memory layout, or compiler-specific behavior.

### Test suite

| Test | Threads | Iterations | What it tests |
|---|---|---|---|
| `atomic_shared_ptr_test` | 4 | 400K/thread | `atomic_shared_ptr` scan/CAS/swap/reset under concurrent access; verifies object lifecycle balance (construction = destruction), use_count correctness |
| `atomic_queue_test` | 4 | 100K pushes/thread | Lock-free queues (`atomic_queue`, `atomic_pointer_queue`, `atomic_queue_reserved`); verifies no lost/phantom elements, totals match |
| `atomic_scoped_ptr_test` | 4 | 1M/thread | Atomic unique-pointer swaps and resets; verifies no leaks (destructor/constructor count balance) |
| `mutex_test` | 8 | 100K lock/unlock/thread | Mutual exclusion correctness; verifies critical section invariant (`g_cnt1` stays balanced) |
| `transaction_test` | 4 | 2,500/thread | STM with tree node operations; verifies **snapshot consistency** (`gn2 <= gn3` invariant across concurrent commits), insert/release under transactions, final sums |
| `transaction_negotiation_test` | 6 (2 slow + 4 fast) | 10 slow, 5M fast | STM fairness under asymmetric contention; verifies slow threads complete despite fast threads (no starvation) |
| `transaction_dynamic_node_test` | 4 | 2,500/thread | STM with dynamic node creation/deletion (ComplexNode subtrees); verifies tree consistency during concurrent restructuring |

### Build & run

```bash
cd tests
# Via qmake (Qt Creator)
qmake tests.pro && make

# Via GNU Make
make -f Makefile.tests check   # builds and runs all tests
```

### Relationship to formal verification

| Property | Stress tests | GenMC | TLA+ |
|---|---|---|---|
| Memory ordering correctness | Probabilistic (timing-dependent) | **Exhaustive** (RC11 model) | N/A (seq. consistency) |
| Protocol logic (refcount) | Probabilistic | Assertion-based | **Exhaustive** (115.7M states) |
| STM commit correctness | `gn2 <= gn3` invariant | N/A | **Exhaustive** (109.9M states) |
| Bundle/unbundle protocol | Implicit (via STM tests) | N/A | **Exhaustive** (622M states) |
| Fairness / starvation | `transaction_negotiation_test` | N/A | Not modeled |
| Real compiler/hardware bugs | **Yes** (actual binary) | No (abstract model) | No (abstract model) |

The stress tests and formal methods are complementary: stress tests catch implementation-level
bugs (compiler optimizations, platform-specific alignment, actual memory ordering on hardware)
while model checking guarantees protocol correctness across **all** possible interleavings
within the modeled abstraction.

---

## Paper-ready citations

- **GenMC:** Kokologiannakis, M., Raad, A., & Vafeiadis, V. "Model Checking for Weakly Consistent Libraries." PLDI 2019.
- **RC11:** Lahav, O., Vafeiadis, V., Kang, J., Hur, C.-K., & Dreyer, D. "Repairing Sequential Consistency in C/C++11." PLDI 2017.
- **TLA+:** Lamport, L. "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers." Addison-Wesley, 2002.
- **TLC:** Yu, Y., Manolios, P., & Lamport, L. "Model Checking TLA+ Specifications." CHARME 1999.
