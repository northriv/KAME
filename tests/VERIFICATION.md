# Formal Verification of KAME's Lock-Free STM Primitives

## Overview

Two complementary verification approaches:

| Layer | Tool | Target | What it verifies |
|---|---|---|---|
| Memory model | GenMC v0.16.1 (RC11) | `atomic_shared_ptr` | `memory_order_relaxed` / `acq_rel` safety under weak memory |
| Protocol logic | TLA+ (TLC) | bundle/unbundle | Multi-phase CAS protocol correctness under sequential consistency |

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

## 2. TLA+: bundle/unbundle Protocol Verification

**Directory:** `tests/tla_bundle/`

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
| A: 2-level | 2 | N/A* | N/A* | 1,248,580 | 60 | 3s | **Pass (complete)** |
| B: 3-level | 1 | 3 | 2 | 597,168 | 84 | 1s | **Pass (complete)** |
| B: 3-level | 2 | 3 | 1 | 413,760,959+ | 75+ | ~60min | **Pass (disk full, no violations)** |

*Model A used CONSTRAINT StateConstraint with MaxPayload=2 instead of modular arithmetic.

### Build & run

```bash
cd tests/tla_bundle

# One-time: download TLA+ tools
curl -sL -o tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Java required
brew install java
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"

# Run Model A (2-level, fast):
# Edit MC.cfg to select model, then:
java -XX:+UseParallelGC -jar tla2tools.jar -config MC.cfg MC.tla -workers auto

# For Model B with large state space, increase heap:
java -XX:+UseParallelGC -Xmx32g -jar tla2tools.jar -config MC.cfg MC.tla -workers auto
```

### MC.cfg for Model A (2-level tree)

Use the `BundleUnbundle_2level.tla` spec with:

```
SPECIFICATION Spec

CONSTANTS
    Threads = {t1, t2}
    Parent  = P
    Child1  = C1
    Child2  = C2
    Null    = NULL
    MaxPayload = 2

CONSTRAINT
    StateConstraint

INVARIANTS
    Safety
```

### MC.cfg for Model B (3-level tree)

Use the `BundleUnbundle.tla` spec with:

```
SPECIFICATION Spec

CONSTANTS
    Threads = {t1, t2}
    Grand   = G
    Parent  = P
    Child1  = C1
    Child2  = C2
    Null    = NULL
    MaxSerial  = 3
    MaxPayload = 1

INVARIANTS
    Safety
```

For machines with large disk/RAM, increase `MaxPayload` to 2 or `MaxSerial` to 4 for
deeper exploration. Estimated state space: ~1B states for MaxPayload=2, MaxSerial=3.

### Liveness

Lock-free progress (`~>`) was tested but violated — this is expected. The protocol is
lock-free (system-wide progress) but not wait-free (per-thread). The `negotiate()` backoff
mechanism, which provides fairness in the real implementation, is not modeled.

---

## Paper-ready citations

- **GenMC:** Kokologiannakis, M., Raad, A., & Vafeiadis, V. "Model Checking for Weakly Consistent Libraries." PLDI 2019.
- **RC11:** Lahav, O., Vafeiadis, V., Kang, J., Hur, C.-K., & Dreyer, D. "Repairing Sequential Consistency in C/C++11." PLDI 2017.
- **TLA+:** Lamport, L. "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers." Addison-Wesley, 2002.
- **TLC:** Yu, Y., Manolios, P., & Lamport, L. "Model Checking TLA+ Specifications." CHARME 1999.
