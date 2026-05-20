# Formal Verification of KAME's Lock-Free STM Primitives

## Overview

Three complementary verification approaches covering the full stack.

- **Layer 0 (informal)**: C11 RC11 memory model — verified by GenMC.
- **Layer 1 (TLA+)**: `atomic_shared_ptr` tagged-pointer primitive **and**
  commit-style primitives (`compareAndSet_impl_`, `scoped_atomic_view`,
  drain `release_tag_ref_`) — merged: the modern API exposes commit-level
  operations directly, subsuming the formerly separate "stm_commit" layer.
- **Layer 2 (TLA+)**: 2/3-level bundle/unbundle with LL-free negotiate,
  plus hard-link topologies (sibling parents and root-with-intermediate
  self-collision).

| Layer | Tool | Target | What it verifies |
|---|---|---|---|
| 0 (informal) | GenMC v0.17 (RC11) | `atomic_shared_ptr` | `memory_order_relaxed` / `acq_rel` safety under weak memory |
| 1 (TLA+) | TLC | `atomic_shared_ptr` | Tagged-pointer protocol + drain release + scoped_atomic_view RAII (TagHeld + CAS set + dtor). Safety only — no liveness (livelock-free is a Layer 2 property) |
| 2 (TLA+) | TLC | 2/3-level bundle/unbundle | Multi-phase CAS + LL-free priority for Parent→Children / Grand→Parent→Children; recursive inner bundle; static + dynamic; **safety + liveness** |
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
| `cds_test_swap.c` | `local_shared_ptr::swap(atomic_shared_ptr&)` under contention | 2 |
| `cds_test_multi_cas_excess.c` | Drain `release_tag_ref_(pref, T)` with `cas_rcnt` parameter | 3 |
| `cds_test_swap_excess.c` | Swap with drain release | 2 |
| `cds_test_cas_excess.c` | CAS with drain release excess undo | 2 |
| `cds_test_cas_noacquire.c` | `compareAndSet_impl_<NOSWAP=true>` no-acquire optimization | 2 |
| `cds_test_scoped_weak.c` | `scoped_atomic_view` + `compareAndSetWeak` race (Layer 1 SCOPED path) | 2 |

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

All 9 tests pass under RC11 (`make run` in `tests/cds_atomic_shared_ptr/`):

| Test | Complete executions | Blocked executions | Wall time | Result |
|---|---|---|---|---|
| 1: load_shared / release_tag_ref safety | 5,757 | 6,428 | 1s | **Pass** |
| 2: load + compareAndSwap_ race | 240 | 7 | 0.06s | **Pass** |
| 3: multiple compareAndSwap_ race | 464,164 | 705,296 | ~270s | **Pass** |
| 4: swap + local_reset safety | 4 | 0 | 0.1s | **Pass** |
| 5: multi CAS with release_tag_ref drain | (large) | (large) | (varies) | **Pass** |
| 6: CAS+load with release_tag_ref drain | (large) | (large) | (varies) | **Pass** |
| 7: swap with release_tag_ref cas_rcnt | 120,118 | 401,606 | 117s | **Pass** |
| 8: compareAndSet_ (NOSWAP=true) no-acquire | 74 | 3 | 0.13s | **Pass** |
| 9: scoped_atomic_view + compareAndSetWeak race | 85 | 12 | 0.04s | **Pass** |

"Blocked executions" are partial executions pruned by GenMC's DPOR algorithm as redundant.

### TLA+-derived GenMC tests (`tests/tlaplus/test_*.c`)

C11 programs **mechanically translated** from the TLA+ specifications, with each
TLA+ atomic step corresponding 1:1 to a C atomic operation:

| File | Source spec | Status |
|---|---|---|
| `test_atomic_shared_ptr.c` | `atomic_shared_ptr.tla` (Layer 1, core) | Pass |
| `test_scoped_atomic_view.c` | `atomic_shared_ptr.tla` (Layer 1, scope) | Pass — 96 executions, 0.28s |
| `test_stm_commit.c` | (legacy stm_commit layer) | Pass |
| `test_bundle_2level.c`, `test_bundle_2level_LLfree.c` | `BundleUnbundle_2level*.tla` | Pass |
| `test_bundle_3level.c`, `test_bundle_3level_LLfree.c` | `BundleUnbundle_3level*.tla` | Pass |

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

## 2. TLA+ Layer 1: `atomic_shared_ptr` + Commit Primitives Verification

**Directory:** `tests/tlaplus/`
**Spec:** `atomic_shared_ptr.tla`

### What it tests

The full Layer 1 vocabulary of `kame/atomic_smart_ptr.h` under sequential
consistency.  Complements GenMC (which checks memory ordering) by exhaustively
exploring all thread interleavings of:

- the tagged-pointer reference-counting protocol
  (`acquire_tag_ref_`, `load_shared_`, `release_tag_ref_`),
- the **modern drain release pattern**
  (`release_tag_ref_(pref, added_global_rcnt)` with `cas_rcnt` parameter),
- the **modern bulk-transfer `load_shared_`**
  (`pref->refcnt.fetch_add(rcnt)` + drain), which steals concurrent threads'
  tag IOUs and shifts contention from `m_ref` to `refcnt`,
- the unified **`compareAndSet_impl_<OldrT, NewrT, false (no WEAK), RETAIN_NEWR>`**
  template (covering Set / Swap variants),
- **`scoped_atomic_view<T>` RAII lifecycle** (TagHeld state):
  acquire → CAS set (SCOPED variant with step4 = +T, fetch_sub(2) on success) → dtor.

### Modeled operations

| Operation | C++ source | Key detail |
|---|---|---|
| `acquire_tag_ref_()` | `atomic_smart_ptr.h:1058-1108` | Single atomic load of `m_ref` + CAS to +1 local tag |
| `load_shared_()` (bulk) | `atomic_smart_ptr.h:1116-1128` | `fetch_add(rcnt)` to global + drain `release_tag_ref_(pref, rcnt)` |
| `release_tag_ref_(pref, T)` | `atomic_smart_ptr.h:1158-1206` | Drain `min(local_rc, T)` tags in one CAS + fetch_sub the excess |
| `compareAndSwap_()` (legacy) | `atomic_smart_ptr.h:550-650` | 6-phase: pre-inc, acquire, check, transfer, CAS, cleanup/undo |
| `local_shared_ptr::swap(asp&)` | `atomic_smart_ptr.h:628-649` | Like CAS but unconditional and hold-transfer |
| `compareAndSet_impl_<SCOPED>` | `atomic_smart_ptr.h:1240-1450` | No acquire (scope holds +1); step4 = +T (full); fetch_sub(2) on success |
| `scoped_atomic_view` ctor | `atomic_smart_ptr.h:598-700` | Acquire → TagHeld |
| `scoped_atomic_view` dtor | `atomic_smart_ptr.h:730-845` | `release_tag_ref_(pref, 1)` if TagHeld |
| `local_shared_ptr::reset()` | `atomic_smart_ptr.h:433-444` | `fetch_sub(1, acq_rel)` + delete check |

### Key modeling decisions

1. **WEAK CAS spurious failure NOT modelled** — `compare_exchange_strong` semantics
   used throughout.  Sound for safety: weak-CAS behaviors are a superset of
   strong-CAS behaviors, and spurious failures exercise the same undo paths
   covered by genuine failures.  The weak case is empirically validated by
   GenMC RC11 (`cds_test_scoped_weak.c`).
2. **ABA-by-recycling NOT modelled** — prevented by the shared-pointer
   ownership contract (`oldr->refcnt >= 1` keeps oldr alive); not a
   structural property of the tagged-pointer protocol.  See spec header.
3. **`scoped_atomic_view` simplified to 2 states** — `"none"` and `"tagheld"`.
   The `"empty"` and `"owned"` C++ states collapse to `"none"` in this model
   (no observable behavior depends on the distinction at Layer 1).
4. **Drain modelled by extending the existing release CAS** — `cas_rcnt`,
   `added_global_rcnt`, and `drained` parameters generalise the 1-tag
   release to arbitrary K-tag release.

### Verified invariants (10)

| Invariant | Description |
|---|---|
| `TypeOK` | Type invariant |
| `MemorySafety` | While a thread holds a tag/ref on pref, pref is not freed |
| `NoUseAfterFree` | Objects held by `local_shared_ptr`s are not freed |
| `GlobalRCNonNeg` | Global refcount >= 0 for live objects |
| `FreedImpliesZeroRC` | Freed objects have `global_rc = 0` |
| `InstalledNotFreed` | Object currently in `atomic_shared_ptr.m_ref` is not freed |
| `LocalRCBounded` | `local_rc <= 2 * |Threads|` (each thread: 1 scope tag + 1 in-flight acquire) |
| `QuiescentCheck` | All-threads-idle implies `freed[o] = (global_rc[o] = 0)` |
| `TerminalCheck` | All-budgets-exhausted + no held refs + scope="none" → ptr has rc=1, others freed |
| `ScopeConsistent` | `scope_state` and `scope_pref` agree; tagheld implies pref not freed |

### Liveness

**Liveness `<>AllDone` is INTENTIONALLY VIOLATED** at Layer 1.  TLC finds a
lasso where two threads enter mutual CAS retry (`cas_retry` ctx → `cas_acquire` →
`atr_cas` → ...) — internal CAS retries do not consume `iterBudget`, so an
adversarial scheduler can sustain the loop indefinitely.  This is the expected
behavior: livelock-freedom is supplied by Layer 2's priority older-wins
(LL-free negotiate) mechanism, not by Layer 1.

### Results

All configurations enable `SYMMETRY AllSymmetry` (Threads ∪ Objects permutable)
except 1-thread and liveness configs.

| Config | Threads | MaxOps | Ops | Distinct states | Depth | Time | Result |
|---|---|---|---|---|---|---|---|
| `atomic_shared_ptr_1thr_mc.cfg` | 1 | 3 | load+CAS+swap+scope | 4,014 | 33 | <1s | **Pass** (complete) |
| `atomic_shared_ptr_small_mc.cfg` | 2 | 2 | load+CAS+scope | 613,990 | 69 | 12s | **Pass** (complete) |
| `atomic_shared_ptr_mc.cfg` | 2 | 3 | load+CAS+scope | (subset of all_mc.cfg) | — | — | superseded |
| `atomic_shared_ptr_all_mc.cfg` | 2 | 3 | load+CAS+swap+scope (full) | **66,537,058** | 84 | 1h 30min | **Pass** (complete) |
| `atomic_shared_ptr_3thr_cas_mc.cfg` | 3 | 2 | load+CAS+scope | — | — | — | (running on local) |
| `atomic_shared_ptr_live_mc.cfg` | 2 | 2 | `<>AllDone` liveness | (lasso) | — | 40s | **Fail (expected)** — confirms Layer 1 has no LL-freedom |

### Key findings

- **All modelled safety invariants hold** under exhaustive 1-thread (4k
  states) and 2-thread MaxOps=2 (614k states) exploration — no memory safety
  violations, no refcount leaks, no use-after-free.
- **Drain release preserves refcount balance** under arbitrary `cas_rcnt`
  values: excess undo (`fetch_sub(added - drained)`) keeps `global_rc`
  consistent whether the drain succeeds or falls through.
- **Scope tag CAS-consume invariant holds**: SCOPED CAS's `fetch_sub(2)` on
  success uniformly absorbs both `m_ref`'s release and scope's tag-share,
  regardless of whether the scope was in the ABSORBED or DRAINED logical
  state at CAS time.
- **Layer 1 has no liveness guarantee** — formally confirmed via the
  `EventuallyAllDone` counter-example, motivating the need for Layer 2's
  priority older-wins mechanism.

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

Same protocol as the 2-level Layer 2 (§3) extended to a 3-level tree
(Grand → Parent → {Child1, Child2}).  Additionally verifies recursive
bundling (snapshot of Grand triggers inner bundle of Parent) and
multi-level unbundle walk.

### Additional coverage vs the 2-level spec

- **Recursive inner bundle**: `InnerPhase2/3/4` model the inner `bundle()` call when
  `snapshot(Grand)` encounters Parent with `missing=TRUE`
- **Multi-level unbundle walk**: `commit(Child)` when Child is bundled 2 levels deep
  (Child → Parent → Grand); walk traverses `bundledBy` chain up to Grand
- **`BundleChainValid`** / **`BundledByCorrect`** invariants: structural correctness of
  the 3-level `bundledBy` chain

### Verified invariants (adds to the 2-level set)

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

### Liveness (Layer 2 only)

**Layer 2 is lock-free + livelock-free**, formally proven via the
`EventuallyAllDone` PROPERTY.  Priority gating (TagOlder older-wins
arbitration) makes unbounded retries structurally impossible: serials
increment monotonically, the state graph is acyclic, and TLC terminates
without `CONSTRAINT`.  See `tests/tlaplus/doc/proof_semantics.md` §2–§4
for the full argument.

**Layer 1 is NOT livelock-free** — see §2 above for the counter-example.
Liveness is supplied by the Layer 2 priority mechanism, not by the bare
tagged-pointer protocol.

Wait-free is not claimed at either layer — CAS-retry-based with fairness
(`WF_vars`) required for per-thread liveness.  See `proof_semantics.md` §7.

---

## 5. TLA+ Layer 2, Hard-link Verification (2-parent + 1-child topologies)

**Directory:** `tests/tlaplus/`
**Specs:**
- `BundleUnbundle_hardlink_dynamic.tla` — sibling-parents variant (Parent1 || Parent2 share DynChild1)
- `BundleUnbundle_hardlink_self_collision.tla` — self-collision variant (R-root + intermediate parent A, both directly parent of C)

### Background

The "hard-link" case — one child node referenced from two distinct parents
— had been called out informally in `CLAUDE.md` / `README.md` as a
"transactions may always fail and retry" liveness caveat.  Production
reports of low-frequency dyn_node aborts ("30/30 abort on another env")
prompted formal modelling of the protocol under hard-link topologies.

Two complementary topologies are modelled:

| Spec | Topology | Bug surface |
|---|---|---|
| `_hardlink_dynamic` | Parent1 ‖ Parent2 → DynChild1 (sibling parents) | migration race when one parent's bundle pulls the child from the other parent's `sub[]` |
| `_hardlink_self_collision` | R → A → C and R → C (root is also direct parent) | bundle(R) walks both R→C and R→A→C; the `is_bundle_root` Phase 4 `m_missing=false` override interacts with the doubled path |

### Self-collision: bug repro and fix simulation

The self-collision spec (`_hardlink_self_collision`) reproduces a real
race captured in the production trace and confirms a proposed fix.

**Bug surface:** during `bundle(R)` Phase 2 (R missing=TRUE, child sub-packets
collected), a peer thread's `insert(R, C)` (hard-link registration) CAS
of R may execute, overwriting R's wrapper with `missing=FALSE`.  The
bundling thread's Phase 4 CAS then fails; on retry, Phase 1 collects C
via a now-stale `bundledBy=A` branch and writes `R.sub[A].sub[C] = Null`,
losing the packet.  `SnapshotConsistency` (mirroring C++
`Packet::checkConsistensy` at `transaction_impl.h:870-871`) is violated.

**Proposed fix:** `BundlePhase3` should only CAS-tag child wrappers
whose packets actually *move* into the parent's `sub[]`.  Hard-link
references — where `parent.sub[c] = Null` indicates the packet lives
elsewhere — must be skipped:

```cpp
// pseudo-diff in bundle() Phase 3
for (child in subnodes) {
    if (local.subpackets[child] == nullptr) continue;  // ← NEW
    // existing CAS to BundledRef(this)
}
```

A naive alternative — gating `insert(R, C)` on `~R.missing` (waiting for
in-flight bundle to clear) — was rejected: it breaks lock-freedom and
risks deadlock at 3+ threads.  The Phase 3 skip lives entirely inside
the bundle protocol and needs no external coordination.

### Impact on non-hard-link models / tests

The fix only triggers when `local.subpackets[child] == nullptr`.  In
the non-hard-link 2level/3level dynamic models and the existing C++
tests (`dyn_node_test`, `3level_mixed_test`, `payload_integrity_*`),
every inserted child's packet lives in exactly one parent's `sub[]` —
no `Null` sub-slot occurs in steady state — so the fix is a no-op.

### Verified invariants

| Invariant | Description |
|---|---|
| `SnapshotConsistency` | mirrors `Packet::checkConsistensy` — a `Null` sub-slot is consistent only if reachable via `reverseLookup` (hard-link path) |
| `HardlinkExclusive` | at most one parent's `sub[]` holds the child's packet (no duplicate homing) |
| `BundleRefConsistency` | `child.bundledBy` parent has priority and either holds the packet or the child wrapper carries an `InsertedRef` |
| `NoPriorityLoss`, `NoMissingHole`, `MissingPropagation` | structural sanity |
| `DebugRetryBound` | per-thread bundle-Phase-1 entries < 10 (catches runaway retry — Lamport-bound equivalent for this serial-less model) |

### Results

`_hardlink_dynamic` (sibling parents, superfine):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 1-thread, MaxCommits=1 | 1 | 7 | **Pass** |
| 2-thread, MaxCommits=1 | 2 | 62 | **Pass** |
| 2-thread, MaxCommits=2 | 2 | 703 | **Pass** |

`_hardlink_self_collision` (R-A-C, before fix):

| Config | Result |
|---|---|
| 2-thread | **FAIL** — `SnapshotConsistency` violated at 376 states / 222 distinct |

`_hardlink_self_collision` (after Phase 3 skip-Null fix):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 2-thread | 2 | 270 | **Pass** + liveness |
| 3-thread | 3 | 6,396 | **Pass** + liveness |

`_hardlink_4node` (R + A + B + shared C, production-race repro
with Phase 4 reachability gating + outer-retry semantics):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 2-thread (production-race, no fix) | 2 | 97 | **FAIL** — SnapshotConsistency violated |
| 2-thread (Phase 4 reachability gating + DISTURBED retry) | 2 | 13,056 | **Pass** + liveness under `SF(MigrateCToA)` |

### First per-action Strong Fairness in the KAME TLA+ corpus

The `_hardlink_4node` (with the C++ fix mirrored) is the first KAME
TLA+ spec to use **per-action Strong Fairness** rather than the
blanket `WF_vars(NextStep)`:

```tla
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads : SF_vars(MigrateCToA(t))
```

Rationale.  With the production-faithful retry semantics (Phase 4 on
reachability failure returns to `bundle_phase1` — matching the C++
`BundledStatus::DISTURBED` return out to `snapshot()`'s outer retry
loop), the model admits executions where the bundling thread retries
arbitrarily many times.  The retry only terminates after the peer's
release-step-2 (`MigrateCToA` in the model) fires, restoring
`A.sub[C]` to a reachable state.

`WF_vars(NextStep)` is too weak here.  Weak fairness on a
disjunction guarantees "some enabled action fires" — an infinite
sequence firing only `BundlePhase1A/B/2/4` (= bundle retries)
satisfies it, even though `MigrateCToA` is continuously enabled in
parallel.  Such a trace violates the production fairness assumption
(LL-free negotiate + OS-level fair scheduling), but TLC's liveness
checker accepts it.

`SF_vars(MigrateCToA(t))` — strong fairness on the specific action —
excludes those traces: every infinite execution must include
`MigrateCToA(t)` firing whenever it is enabled infinitely often.
After it fires, the next bundle attempt sees the reachable state and
finalizes cleanly.

All earlier KAME TLA+ specs (`BundleUnbundle_2level_LLfree`,
`_3level_LLfree`, both `_dynamic` variants, the other hardlink
specs, and `atomic_shared_ptr.tla`) terminate without needing
per-action SF — they either bound retries structurally via LL-free
priority gating, or use blanket `WF_vars(NextStep)` for cases
where any-action progress is sufficient.

The hardlink-with-external-migration race is genuinely different:
two threads making independent progress, one thread's forward
progress dependent on the other's specific step.  Production
achieves this via LL-free negotiate priority + scheduler fairness;
the model abstracts both as `SF(MigrateCToA)`.

### DebugRetryBound usage clarification

| Model | DebugRetryBound mode | Rationale |
|---|---|---|
| `_hardlink_self_collision` | INVARIANT (bound 10) | Bundle retries are structurally bounded; INVARIANT catches false-negative liveness misses |
| `_hardlink_external_migration` | CONSTRAINT (bound 5) | Peer race can cause unbounded retries; documented model limitation, CONSTRAINT bounds state space for safety check |
| `_hardlink_4node` (production-faithful) | CONSTRAINT (bound 20) | DISTURBED-style retry is unbounded by construction; CONSTRAINT bounds state space, SF(MigrateCToA) proves liveness analytically |

User feedback 2026-05-20 flagged that CONSTRAINT can mask
false-negative liveness verdicts.  The fix is two-pronged:
1. For models with structurally-bounded retry → use INVARIANT
2. For models with retry bounded only by fair scheduling → use
   CONSTRAINT *with explicit fairness annotation* (SF on the
   progress-making action), so liveness verification under fairness
   is analytically sound within the bound

### Build & run

```bash
cd tests/tlaplus

# Self-collision (after fix) 2-thread (<1 s)
java -XX:+UseParallelGC -Xmx4g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_hardlink_self_collision_2thr_mc.cfg \
  BundleUnbundle_hardlink_self_collision.tla

# 4-node production-race + fix 2-thread (~1 s)
java -XX:+UseParallelGC -Xmx6g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_hardlink_4node_2thr_mc.cfg \
  BundleUnbundle_hardlink_4node.tla
```

To reproduce the original bug, revert the Phase 3 `subpackets[c] == Null`
skip in `BundlePhase3` and `BundlePhase3Fail` (re-add the `CAS C` branch
and the `local[t].cWrapper /= Null` disjunct), or in `_hardlink_4node`
remove the `canFinalize` gating in `BundlePhase4`.

---

## 6. Runtime Stress Tests (C++)

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
| Protocol logic (refcount + drain + scope) | Probabilistic | Assertion-based | **Exhaustive** (Layer 1, this work) |
| STM commit correctness | `gn2 <= gn3` invariant | N/A | **Exhaustive** (Layer 1 + Layer 2) |
| Bundle/unbundle protocol | Implicit (via STM tests) | N/A | **Exhaustive** (Layer 2, 622M states) |
| Livelock-freedom (LL-free) | Asymmetric stress | N/A | **Exhaustive** (Layer 2 only; Layer 1 has counter-example) |
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
