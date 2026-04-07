# Formal Verification of KAME's Lock-Free STM Primitives

## Overview

Three complementary verification approaches covering the full stack:

| Layer | Tool | Target | What it verifies |
|---|---|---|---|
| Memory model | GenMC v0.16.1 (RC11) | `atomic_shared_ptr` | `memory_order_relaxed` / `acq_rel` safety under weak memory |
| Layer 0 (TLA+) | TLC | `atomic_shared_ptr` | Tagged-pointer refcounting protocol (scan/CAS/reset) with ABA + non-atomic reads |
| Layer 1 (TLA+) | TLC | STM commit | Single-node optimistic Snapshot + Write + Commit cycle |
| Layer 2 (TLA+) | TLC | bundle/unbundle | Multi-phase CAS protocol for subtree atomic snapshots |

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

## 2. TLA+ Layer 0: `atomic_shared_ptr` Protocol Verification

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

- **ABA problem does not occur**: Even with object recycling at the same pointer value, the tagged-pointer's local_rc makes CAS distinguish recycled objects from originals. 115.7M states verified with no ABA violation.
- **Non-atomic read is safe**: Inconsistency between the two loads is always caught by the subsequent CAS (which compares the full atomic word).
- **`leave_scan_` fallback branch is essential**: When the pointer is swapped between reserve and leave, the `global_rc--` fallback (lines 526-529) correctly maintains refcount invariants. Without it, refcount leaks or use-after-free would occur.

---

## 3. TLA+ Layer 1: STM Commit Protocol Verification

**Directory:** `tests/tlaplus/`  
**Spec:** `stm_commit.tla`

### What it tests

The optimistic concurrency control cycle for single-node transactions from `kame/transaction.h`
and `kame/transaction_impl.h`. Abstracts `atomic_shared_ptr` (verified in Layer 0) as a
correct atomic register.

### Modeled cycle

1. **Snapshot**: Read current `PacketWrapper` via atomic scan (captures `node_val` + `node_serial`)
2. **Write**: Copy-on-write modification of payload (both nondeterministic and deterministic increment)
3. **Commit**: CAS on `m_link` — succeeds only if `(node_val, node_serial)` unchanged since snapshot
4. **Retry**: On CAS failure, take new snapshot and repeat (`iterate_commit` pattern)

### Verified invariants (6)

| Invariant | Description |
|---|---|
| `NoLostUpdate` | If two threads both committed, at least 2 serial increments occurred |
| `CommitSerializes` | Total commits across all threads <= node_serial |
| `SnapshotBeforeCommit` | Each committer's snapshot serial < current serial |
| `WriteReadConsistency` | Last committer's write value is reflected in node_val |
| `ValueBounded` | node_val <= MaxVal |
| `TypeOK` | Type invariant |

### Results

| Threads | MaxVal | MaxSerial | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|---|
| 3 | 3 | 6 | 109,901,200 | 27 | 10min | **Pass (complete)** |

---

## 4. TLA+ Layer 2: bundle/unbundle Protocol Verification

**Directory:** `tests/tlaplus/` (also `tests/tla_bundle/`)

### What it tests

The multi-phase CAS protocol that makes subtrees atomically snapshotable. The specification
models `bundle()`, `unbundle()`, `commit()`, and `snapshot()` from `kame/transaction_impl.h`.

### Two models

#### Model A: 2-level tree (Parent → {Child1, Child2})

Simpler model covering:
- 4-phase bundle protocol (collect → CAS parent → CAS children → finalize)
- Unbundle for commit (CAS parent to mark slot missing → CAS child to restore priority)
- Concurrent snapshot + commit interference
- Single-node commit optimization (adopt new children if payload unchanged)

#### Model B: 3-level tree (Grand → Parent → {Child1, Child2})

Full model additionally covering:
- **Recursive bundling**: `snapshot(Grand)` bundles Parent, which bundles Children
- **Multi-level unbundle**: `commit(Child)` when bundled 2 levels deep (Child→Parent→Grand)
- **`snapshotSupernode()` walk**: traversing `bundledBy` chain up 2 levels
- **`BundledByCorrect` invariant**: `bundledBy` always points to structural parent

### Modeled operations

**Snapshot (triggers bundle if needed):**
1. Read node's linkage
2. If `hasPriority` and not `missing`: fast path (return packet)
3. If `hasPriority` and `missing`: 4-phase bundle:
   - Phase 1: Collect sub-packets from children
   - Phase 2: CAS node's linkage with new packet (missing=TRUE)
   - Phase 3: CAS each child to `BundledRefWrapper` (back-reference to parent)
   - Phase 4: CAS node's linkage with missing=FALSE (finalize)
4. If not `hasPriority` (bundled): retry

**Commit (triggers unbundle if needed):**
1. Read target node's linkage
2. If `hasPriority`: CAS with new payload (fast path)
   - Single-node optimization: if CAS fails but payload unchanged, adopt new children
3. If bundled: unbundle walk
   - 1-level: CAS parent (mark child slot Null, missing=TRUE) → CAS child (restore priority)
   - 2-level (Model B only): CAS grandparent → restore parent → CAS child

### Verified invariants

| Invariant | Description |
|---|---|
| `SnapshotConsistency` | If node has `missing=FALSE`, all sub-packets exist |
| `NoPriorityLoss` | Non-root nodes always have `hasPriority=TRUE` or `bundledBy≠Null` |
| `BundleChainValid` | (Model B) Bundled node's `bundledBy` target is priority or itself bundled |
| `BundledByCorrect` | (Model B) `bundledBy` always points to the structural parent |
| `GrandAlwaysPriority` | (Model B) Root node always has priority |
| `BundleRefConsistency` | (Model A) If child bundled to parent, parent has priority |

### Modular serial arithmetic

Serials and payload versions use modular arithmetic (`% MaxSerial`, `% MaxPayload`) to make
the state space naturally finite without artificial CONSTRAINT cutoffs.

### Results

| Model | Threads | MaxSerial | MaxPayload | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|---|---|
| A: 2-level | 2 | N/A* | 2 | 3,967,507 | 62 | 11s | **Pass (complete)** |
| B: 3-level | 2 | 3 | 1 | 622,118,022 | 167 | 4h 20min | **Pass (complete)** |

*Model A uses CONSTRAINT StateConstraint with MaxPayload=2 instead of modular arithmetic.

### Build & run

```bash
cd tests/tlaplus

# Java required
brew install openjdk
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"

# All TLA+ specs (Layer 0, 1, 2) and tla2tools.jar are in this directory.

# Layer 0: atomic_shared_ptr (fast — 9 min)
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC \
  atomic_shared_ptr -config atomic_shared_ptr_mc.cfg -workers auto

# Layer 1: STM commit (fast — 10 min)
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC \
  stm_commit -config stm_commit_mc.cfg -workers auto

# Layer 2, Model A: 2-level bundle/unbundle (fast — 11s)
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC \
  MC_2level -config MC_2level.cfg -workers auto

# Layer 2, Model B: 3-level bundle/unbundle (heavy — ~4.5 hours, 622M states)
# Use -metadir on a disk with >50GB free space for state files.
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC \
  MC -config MC.cfg -metadir /path/to/large/disk/states -workers auto
```

### Liveness

Lock-free progress (`~>`) was tested but violated — this is expected. The protocol is
lock-free (system-wide progress) but not wait-free (per-thread). The `negotiate()` backoff
mechanism, which provides fairness in the real implementation, is not modeled.

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
