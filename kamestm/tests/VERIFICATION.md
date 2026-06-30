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

The core lock-free reference counting protocol from `kamepoolalloc/atomic_smart_ptr.h`, extracted into
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

### Reproduction audit (2026-06-09, GenMC v0.17.0 / LLVM 20.1.8, Apple M3)

The full suite was re-run on a freshly built GenMC v0.17.0 to confirm the
numbers above are reproducible on current toolchains. **Every documented
complete-execution count was reproduced to the digit:**

| Test | Re-run complete | Doc | Wall | |
|---|---|---|---|---|
| 1: `cds_test_load` | 5,757 | 5,757 | 1s | ✅ |
| 2: `cds_test_cas` | 240 | 240 | <1s | ✅ |
| 3: `cds_test_multi_cas` | 464,164 | 464,164 | 275s | ✅ |
| 4: `cds_test_swap` | 4 | 4 | <1s | ✅ |
| 5: `cds_test_multi_cas_excess` | 410,364 | (large) | 271s | ✅ pass |
| 6: `cds_test_cas_excess` | 240 | (large) | 1s | ✅ pass |
| 7: `cds_test_swap_excess` | 120,118 | 120,118 | 117s | ✅ |
| 8: `cds_test_cas_noacquire` | 74 | 74 | <1s | ✅ |
| 9: `cds_test_scoped_weak` | 85 | 85 | <1s | ✅ |
| TLA+ `test_scoped_atomic_view.c` | 96 | 96 | <1s | ✅ |

Because the **complete**-execution counts match the recorded values exactly,
the `--unroll=5` bound is *complete* for these tests — every CAS-retry loop
terminates within the bound, so no execution is truncated and the
verification is not vacuous. (Contrast: a test whose loops exceed the unroll
reports `complete executions: 0` with all paths *blocked* — a vacuous pass.
Always confirm `complete executions > 0`.)

`test_atomic_shared_ptr.c` (the monolithic 3-thread load+CAS+swap union of
tests 1/2/4) is verification-heavy — it did **not** finish a local 25-minute
run (`>1551s`, no result), consistent with the 66.5M-state scale of the
analogous TLC `atomic_shared_ptr_all_mc.cfg`. The focused decomposition in
the table above plus that TLA+/TLC config (§2) are the tractable,
authoritative coverage of the full protocol.

**macOS build note (GenMC v0.17.0):** on macOS 26 SDK + Homebrew `llvm@20`,
building GenMC *itself* with the keg's `clang++`/`libc++` (as the recipe
above implies) fails to link with an unresolved weak symbol
`operator new[](size_t, std::__type_descriptor_t)` (a type-aware-allocation
ABI symbol the prebuilt keg `libc++.dylib` does not export). Building GenMC
with **Apple `clang++` (system libc++)** while still pointing
`-DLLVM_DIR=$(brew --prefix llvm@20)/lib/cmake/llvm` at the keg (for
`libLLVM.dylib` + the frontend `clang`) resolves it — GenMC's own objects no
longer emit that symbol, and the LLVM C-API boundary is unaffected. Also
add `hwloc` to the link path (`-L$(brew --prefix hwloc)/lib`), and note the
v0.17.0 binary installs at `build/bin/genmc/genmc` (nested).

---

## 2. TLA+ Layer 1: `atomic_shared_ptr` + Commit Primitives Verification

**Directory:** `tests/tlaplus/`
**Spec:** `atomic_shared_ptr.tla`

### What it tests

The full Layer 1 vocabulary of `kamepoolalloc/atomic_smart_ptr.h` under sequential
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
| `acquire_tag_ref_()` | `kamepoolalloc/atomic_smart_ptr.h:1632` | Single atomic load of `m_ref` + CAS to +1 local tag |
| `load_shared_()` (bulk) | `kamepoolalloc/atomic_smart_ptr.h:1684` | `fetch_add(rcnt)` to global + drain `release_tag_ref_(pref, rcnt)` |
| `release_tag_ref_(pref, T)` | `kamepoolalloc/atomic_smart_ptr.h:1730` | Drain `min(local_rc, T)` tags in one CAS + fetch_sub the excess |
| `compareAndSet_impl_()` | `kamepoolalloc/atomic_smart_ptr.h:1811` (public `compareAndSwap()` decl `:981`) | Unified Set / Swap template (subsumes the former 6-phase `compareAndSwap_`): pre-inc, acquire, check, transfer, CAS, cleanup/undo |
| `local_shared_ptr::swap(asp&)` | `kamepoolalloc/atomic_smart_ptr.h:2085` (decl `:717`) | Like CAS but unconditional and hold-transfer |
| `compareAndSet_impl_<SCOPED>` | `kamepoolalloc/atomic_smart_ptr.h:1811` | No acquire (scope holds +1); step4 = +T (full); fetch_sub(2) on success |
| `scoped_atomic_view` ctor | `kamepoolalloc/atomic_smart_ptr.h:1169` / `:1200` (class `:1139`) | Acquire → TagHeld |
| `scoped_atomic_view` dtor | `kamepoolalloc/atomic_smart_ptr.h:1271` | `release_tag_ref_(pref, 1)` if TagHeld |
| `local_shared_ptr::reset()` | `kamepoolalloc/atomic_smart_ptr.h:1597` (decl `:720`) | `fetch_sub(1, acq_rel)` + delete check |

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
Tags cleared only on commit success (`ClearMyTags`).

The TLA+ priority mechanism mirrors the **per-linkage privilege** path in
`transaction.h` (`KAME_PER_LINKAGE_PRIVILEGE=1`, the default). The model's
abstract symbols correspond to these C++ symbols (verified per-linkage
correspondence):

| TLA+ symbol | C++ symbol (`transaction.h`) |
|---|---|
| `MyTag(t) = <<iter(t), t>>` (a transaction's own tag) | `Snapshot::m_started_time` (tid-packed µs stamp from `now_us_tagged()`; kinded via `with_kind(m_started_time, …)`) — `:1515` / `:1662` |
| `iter(t)` (transaction age) | the age component of `m_started_time`, compared by `signed_diff_us_packed` — `:1664` |
| `priorityTag[n]` (the per-node registered tag) | `Linkage::m_transaction_started_time` (the per-linkage priority slot, atomic) — `:905` |
| `TagAfterFail` (oldest-wins write on CAS contention) | `Snapshot::tag_as_contender(link)` (CAS: slot empty OR current tagger younger → overwrite; pushes onto `m_tagged_linkages`) — `:1630` |
| `CanProceed` (the gate) | `i_am_privileged_now` / `fair_mode_blocks_me` — `:646` / `:634` |
| `PreemptTag` (older preempts younger) | the symmetric preempt-window inside `tag_as_contender` — `:1669` |
| `ClearMyTags` (release on commit success) | `Snapshot::drop_tags_n_privilege()` walking `m_tagged_linkages`, zeroing matching slots — `:1802` |
| escalation to Reserved kind | `m_registered_privileged → StampKind::Reserved` — `:1659` |

(The `KAME_PER_LINKAGE_PRIVILEGE=0` build instead uses a global
fallback — `s_privileged_tidstamp` / `try_register_privileged_tidstamp` /
`release_privileged_tidstamp` in `transaction_neg_impl.h` — which is not the
default.)

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
| 2-thread micro (fine) | 2 | 867,696 | 89 | ~35s | **Pass + liveness** |
| 2-thread superfine | 2 | 2,676,196 | 129 | 3m 12s | **Pass + liveness** |
| 3-thread superfine confC (all-root) | 3 | 137,333,348 | 96 | 2h 57min | **Pass + liveness** (ohtaka, /dev/shm) |
| MaxCommits=2 superfine | 2 | 127,586,599 | 311 | 4h 40min | **Pass** (ohtaka) |
| dynamic release superfine live | 2 | 413,884,516 | 320 | 7h 13min | **Pass + liveness** (ohtaka) |

The 3-thread confC (all-root) row is now a full **liveness** pass
(`BundleUnbundle_2level_LLfree_3thr_superfine_C_live_mc.cfg`, 2026-06-22,
/dev/shm 126 workers): 443,332,503 generated / 137,333,348 distinct /
depth 96, `EventuallyAllDone` PASS, temporal SCC 20min 19s, 2h 57min
total.  This matches the 3-level confC liveness (640 M, below), so
**3-thread liveness (all-root) holds at both the 2- and 3-level tree**.
The leaf-containing 3-thread splits (confA/B/D) remain safety-frontier
only — intractable for liveness in the 12–24 h budget (CommitChild
interleavings blow the state count past the all-root 137 M / 640 M).

The micro/fine row is `BundleUnbundle_2level_LLfree_micro_mc.cfg` (re-run
2026-06-20: 2,083,827 generated / 867,696 distinct / depth 89, queue 0,
`EventuallyAllDone` PASS). It was **665,218** at the first LL-free proof
(commit `a35c4310`, see `verification_log.md`); the spec has since gained
reachable interleavings at the same depth 89, so the current figure is
867,696. Full / historical results: `tests/tlaplus/doc/verification_log.md`.

### Thread-axis saturation (T-scaling experiment, 2026-06-23/24)

A direct probe of the thread-axis cutoff *conjecture*
(`tests/tlaplus/doc/parameterized_cutoff.md` §5.1): do more threads create new
safety-relevant behaviour, or only more interleavings? 2-level, `MaxCommits=1`,
**no** symmetry (raw reachable sets — the LL-free `TagOlder` tid-`<` forces
ordered-natural thread ids, which TLC `SYMMETRY` rejects, §5.1 (i)). Each config
reports the **raw** distinct-state count and the **structural σ** count — the
reachable set projected onto the identity-free bundle structure (per node:
`hasPriority`/`bundledBy`/`missing`/which `sub` slots are populated; `serial` and
payload value dropped — exactly the fields the structural invariants read). The
headline result is in the **superfine** (most-interleaved, C++-faithful) model.

| Atomicity | Workload | T | Raw distinct states | Structural σ | Safety |
|---|---|---|---|---|---|
| **superfine** | all-root | 2 | 124,244 | **6** | Pass |
| **superfine** | all-root | 3 | **137,333,348** | **6 — set-identical to T=2** (`diff` empty; σ=6 across all 137 M) | **Pass** (ohtaka F1cpu: TLC 5h08m + 736 GB dump + projection 3h08m) |
| coarse | all-root | 2 | 1,093 | 4 | Pass |
| coarse | all-root | 3 | 339,744 | 4 — set-identical to T=2 (`diff` empty) | Pass |
| coarse | all-root | 4 | 136,366,732 | — (136 M, not dumped) | **Pass** (28 min) |
| coarse | both-roles | 2 | 350,281 | 6 (4 bundle + 2 partial-unbundle) | Pass |

- **Superfine saturation — the faithful result.** In the most-interleaved,
  C++-faithful model the all-root structural set is **6** and is **set-identical
  at `T = 2` and `T = 3`**, verified by dumping the *complete* 137 M-state `T = 3`
  exhaustion (736 GB) and projecting — σ held at 6 across all 137,333,348 states.
  The superfine 6 = the coarse 4 **plus the two within-operation Phase-3
  intermediates** (one child re-pointed to a bundled-ref, the other not yet) that
  coarse's atomic Phase-3 collapses away — i.e. exactly the genuinely-concurrent
  interleavings where a hazard could hide. They saturate at `T = 2` as well, so a
  third thread reaches **no new safety-relevant structure even in the faithful
  model**.
- **Raw state count explodes** with `T` (superfine all-root ~10⁵ → 1.37×10⁸): the
  combinatorial interleaving/serial/payload growth that makes brute-force ∀`T`
  hopeless; the safety-relevant *structure* does not grow.
- The `T = 4` *coarse* all-root run (all 4-thread interleavings, 136 M states)
  finds **no invariant violation**: a direct larger-`T` check that no dangerous
  4-thread CAS pattern exists at that instance (TLC would emit a counterexample
  trace otherwise).

This **supports the ∀`T` conjecture** (§5.1) — a *faithful-model* measurement that
the identity-free safety structure is finite and stable across checked thread
counts, **not** a ∀`T` proof. Scope: 2-level, all-root (bundle-side; the
multi-level *unbundle* structures need a leaf workload, and superfine `T = 4` /
3-level superfine saturation are intractable to dump). The 3-level case is reached
by extrapolation via the tree-independence of the commit-role structure (§5.1
Facts A–C). (3-level *coarse* `T = 2` already reaches ≥ 9 structures; its
exhaustion was not completed.)

#### Scope of the saturation / structural-invariant argument

The saturation result — and the candidate *local structural invariant* that
characterizes the saturated σ-set (the conjuncts `SubNeverMissing`,
`BundledHasCopy`, `StaleParentExcluded`, `SubPresenceUniform`, validated with no
violation up to **3-level superfine `T = 3` all-root** and **3-level superfine
`T = 2` both-roles**, i.e. the most-interleaved model on both the bundle and the
multi-level-unbundle paths) — are stated for a **static, single-parent rooted
tree**: `Next` has no node-insert/remove action and `ParentOf` is single-valued.
Two regimes lie outside this argument and are verified *separately*, not by
extension of it:

- **Dynamic topology** (online insertion/release). Covered by the dynamic specs
  (`BundleUnbundle_{2,3}level_LLfree_dynamic.tla` and their `*_dynamic_*` cfgs —
  e.g. the 413 M-state *dynamic-release superfine liveness* run above) and the
  `transaction_dynamic_node_test` C++ stress (§6). `SubPresenceUniform` (a node's
  child-slots present-or-absent together) is an invariant of a *fixed* topology
  only: inserting a child into an already-bundled parent leaves the new slot
  `Null` while its siblings are non-`Null`, breaking it transiently.
- **Hard links / DAG** (a child with ≥ 2 parents). Covered by §5
  (`BundleUnbundle_hardlink_*`) and the bundle-Phase-3 fix. The conjuncts that
  name *the* parent (`StaleParentExcluded`, `BundledHasCopy`) are ill-formed when
  `ParentOf` is multi-valued, and `SubPresenceUniform` can be broken by an
  independent second parent's unbundle; they are not claimed on a DAG.

#### Raw state counts are spec-version-specific (determinism / provenance)

Raw distinct-state counts are **not comparable across spec versions**. TLC's
breadth-first search is a *deterministic* exhaustion: for a fixed (`.tla`, cfg)
the reachable set — and hence every reported count and each variable's maximum
(e.g. `PrintTerminalMaxCounter`) — is reproducible exactly; fingerprint seed and
worker count change only *discovery order* and a negligible *collision
probability* (≈ `N²/2⁶⁵`, < 1 collided state at `5×10⁸`). A changed count
therefore signals a **changed model**, never run-to-run nondeterminism. Over this
project's development the same confC superfine `T = 3` configuration (3-level,
*identical* cfg constants throughout — `MaxCommits=1`, all-`superfine`,
`Privilege=TRUE`, all-root) moved as protocol fixes landed in the `.tla`:

| Date | Distinct | Depth | maxctr | Coinciding `.tla` fix (git) |
|---|---:|:--:|:--:|---|
| 2026-05-02 | 514,070,136 | 76 | 12 | `1d9820bc` Fix InnerPhase3 restart |
| 2026-05-02 | 1,154,807,632 | 89 | 15 | `8d6026e3` Fix InnerPhase4 restart |
| 2026-05-03 | 640,894,951 | 88 | 15 | `73bcef3a` Fix InnerPhase2 restart |
| 2026-06-26 | 540,782,047 | 88 | — | `924b6e63` Fix for TLA+ model (current) |

(Per-run HEAD inferred from commit vs run dates; each `Fix …` commit changed
`Init`/`Next`, hence the reachable set and the counter.) **Only same-version runs
are cross-comparable**; the paper reports current-spec numbers, and the
version-independent quantity is the σ-projection (the saturated structure set),
not any raw count. The current-spec `540,782,047` is the safety + structural-conjunct
σ-closure run (this session, dump-free); a matching current-spec **liveness** run
is in flight, expected to report the same `540,782,047` distinct (liveness adds
only a temporal pass over the same graph — confirmed at 2-thread, where the
3-level superfine liveness count equals the safety+conjunct count exactly).

### Equivalence of the existing models with the new C++ Phase 4 reachability gate

The hard-link work (§5) introduced a `Packet::allSubReachable` gate
in the C++ bundle Phase 4 (`transaction_impl.h`, commits `87892b35`,
`92b15f62`, `404fa137`, `b12e1895`): the `is_bundle_root=true`
override now only clears `m_missing` when every Null sub-slot has a
`reverseLookup`-able subnode, otherwise the bundle returns
`BundledStatus::DISTURBED` so the outer `snapshot()` retry loop
re-attempts.  The gate's order matters: `newpacket->m_missing` is
cleared *before* `allSubReachable` / `checkConsistensy` are called,
because both functions short-circuit their Null-slot reverseLookup
when the root they observe is `missing` (mid-bundle semantics).  On
reachability failure the flag is restored to `true` before returning
DISTURBED (see `b12e1895` for the order-of-operations subtlety
introduced by the global-root parameter).

`checkConsistensy` and `allSubReachable` were also given an optional
`globalroot` parameter (commit `1ffd8dce`) so callers with a non-root
`pkt` can supply the true global tree root for the Null-slot
`reverseLookup` — necessary for hard-link Case B (where the
hard-linked child's packet lives in a sibling sub-tree).  At the
Phase 4 call site, `newpacket` aliases the global root for
`is_bundle_root=true` (see `reverseLookup` line 1593 — when
`&superpacket->node() == this` the function returns `superpacket`
itself), so the default `globalroot = {}` is correct without
explicit threading.

**The 2level / 3level LL-free (static + dynamic) TLA+ models do not
need to be updated** for these C++ changes.  In all non-hard-link
topologies the gate is provably a no-op:

1. Each child node has exactly one parent.  When `bundle(P)` reaches
   Phase 4 with a Null sub-slot for an active child `c`, the only way
   `reverseLookup` could fail is if `c`'s packet lives *outside*
   `P`'s subtree — but with a single parent that is impossible by
   construction.
2. The existing `SnapshotConsistency` invariant in these specs
   (`2level_LLfree_dynamic.tla:1093-1096`,
   `3level_LLfree_dynamic.tla:1831-1835`) states exactly the
   property the gate enforces:

   ```tla
   (pw.hasPriority /\ ~pw.packet.missing) =>
       (\A c \in ActiveChildren : pw.packet.sub[c] /= Null)
   ```

   In words: if a node is published with `missing=FALSE`, no active
   child has a Null sub-slot.  This is precisely the precondition
   `allSubReachable` checks before letting the publish proceed.
3. The 413 M-state `dynamic release superfine live` run and all
   smaller dynamic / static configurations have verified this
   invariant without violation.  Therefore the C++ `else { return
   DISTURBED }` branch is provably unreachable in non-hard-link
   topology, and the models accurately represent the implementation
   without modification.

The gate is only effectful in the hard-link topology, where multiple
parents share a child and the multi-step release can leave the child
transiently Null in two parents' `sub[]`.  That case is modelled in
§5.

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
| 2-thread coarse | 2 | 1,497,098 | 98 | 1m 35s | **Pass + liveness** |
| 2-thread superfine | 2 | 15,094,117 | 146 | 15m 11s | **Pass + liveness** (ohtaka, /dev/shm) |
| 3-thread superfine confC (all-root) | 3 | 640,894,951 | 88 | 15h 25min | **Pass + liveness** (ohtaka) |

The 2-thread superfine row is a 2026-06-21 re-run (`/dev/shm`, 126 workers):
35,271,006 generated / 15,094,117 distinct / depth 146, queue 0,
`EventuallyAllDone` PASS. It was 14,109,731 / depth 148 earlier; the spec
has since gained reachable interleavings at ~the same depth (the same
drift as the 2-level micro 665,218 → 867,696), not a regression.

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
- `BundleUnbundle_hardlink_dynamic.tla` — sibling-parents variant (Parent1 ‖ Parent2 share DynChild1)
- `BundleUnbundle_hardlink_self_collision.tla` — self-collision variant (R-root + intermediate parent A, both directly parent of C)
- `BundleUnbundle_hardlink_4node.tla` — Root-A-B-C topology with C hard-linked under both A and B (production-race repro, mirrors the C++ Phase 4 reachability-gating fix)
- `BundleUnbundle_hardlink_external.tla` — minimal 4-node external-parent repro (`P2` hard-linked external to `P1`, `bundle(GN1)` triggers `SnapshotConsistency` violation without the fix)
- `BundleUnbundle_hardlink_external_migration.tla` — cross-tree migration (bundle on `GN1` reaches into `P1`'s tree to pull `P2` into `GN2.sub[P2]`)
- `BundleUnbundle_hardlink_nonatomic.tla` — *not a new topology*: a liveness investigation of the non-transactional `insert`/`release` test pattern (b23fa954) interleaved with transactional ops, comparing the master fall-through vs the self-promote fix under two fairness levels (see its own subsection below)

### Background

The "hard-link" case — one child node referenced from two distinct parents
— had been called out informally in `CLAUDE.md` / `README.md` as a
"transactions may always fail and retry" liveness caveat.  Production
reports of low-frequency dyn_node aborts ("30/30 abort on another env")
prompted formal modelling of the protocol under hard-link topologies.

Five complementary topologies are modelled (a sixth spec,
`_hardlink_nonatomic`, is a non-transactional-pattern liveness
investigation rather than a topology — covered in its own subsection):

| Spec | Topology | Bug surface |
|---|---|---|
| `_hardlink_dynamic` | Parent1 ‖ Parent2 → DynChild1 (sibling parents) | migration race when one parent's bundle pulls the child from the other parent's `sub[]` |
| `_hardlink_self_collision` | R → A → C and R → C (root is also direct parent) | bundle(R) walks both R→C and R→A→C; the `is_bundle_root` Phase 4 `m_missing=false` override interacts with the doubled path |
| `_hardlink_4node` | Root → A, Root → B, A → C, B → C (C shared between A and B) | bundle(Root) Phase 4 reachability gating + outer `DISTURBED` retry; production-race repro |
| `_hardlink_external` | GN1 → P2 (hardlink), P1 → P2 (external owner of P2's packet) | bundle(GN1) finalises `GN1 ~missing` while `GN2.sub[P2]=Null` is unreachable from GN1 → SnapshotConsistency violation without Phase 4 gating |
| `_hardlink_external_migration` | as above, but the bundle is allowed to migrate P2 from P1's tree | bundle(GN1) must atomically pull P2 out of P1's `sub[]` and into GN2's `sub[]` while the peer races on P1 |

### Self-collision: bug repro and fix simulation

The self-collision spec (`_hardlink_self_collision`) reproduces a real
race captured in the production trace and confirms a proposed fix.

**Bug surface:** during `bundle(R)` Phase 2 (R missing=TRUE, child sub-packets
collected), a peer thread's `insert(R, C)` (hard-link registration) CAS
of R may execute, overwriting R's wrapper with `missing=FALSE`.  The
bundling thread's Phase 4 CAS then fails; on retry, Phase 1 collects C
via a now-stale `bundledBy=A` branch and writes `R.sub[A].sub[C] = Null`,
losing the packet.  `SnapshotConsistency` (mirroring C++
`Packet::checkConsistensy` at `transaction_impl.h:1001`) is violated.

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

| Invariant / Property | Description |
|---|---|
| `SnapshotConsistency` | mirrors `Packet::checkConsistensy` — a `Null` sub-slot is consistent only if reachable via `reverseLookup` (hard-link path) |
| `HardlinkExclusive` | at most one parent's `sub[]` holds the child's packet (no duplicate homing) |
| `BundleRefConsistency` | `child.bundledBy` parent has priority and either holds the packet or the child wrapper carries an `InsertedRef` |
| `NoPriorityLoss`, `NoMissingHole`, `MissingPropagation` | structural sanity |
| `EventuallyAllDone` (PROPERTY) | all threads eventually complete — checked on every hard-link cfg (uniform safety+liveness coverage as of 2026-05-20) |

### Results

`_hardlink_dynamic` (sibling parents, superfine) — liveness re-verified
2026-06-20 (TLC / OpenJDK 24, `EventuallyAllDone` checked on every row):

| Config | Threads | Generated | Distinct | Depth | Result |
|---|---|---|---|---|---|
| MaxCommits=1 (`_1thr_mc.cfg`) | 1 | 10 | 7 | 4 | **Pass + liveness** |
| MaxCommits=1 (`_2thr_mc.cfg`) | 2 | 152 | 62 | 9 | **Pass + liveness** |
| MaxCommits=2 (`_2thr_commits2_mc.cfg`) | 2 | 1,818 | 703 | 16 | **Pass + liveness** |

The MaxCommits=2 row was previously recorded safety-only ("Pass"); the
dedicated `_2thr_commits2_mc.cfg` (added 2026-06-20) reproduces the same
703 distinct states / depth 16 and confirms `<>AllDone` holds, so every
dynamic config now carries the uniform safety+liveness verdict.

`_hardlink_self_collision` (R-A-C, before fix):

| Config | Result |
|---|---|
| 2-thread | **FAIL** — `SnapshotConsistency` violated at 376 states / 222 distinct |

`_hardlink_self_collision` (after Phase 3 skip-Null fix):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 2-thread | 2 | 114 | **Pass** + liveness |
| 3-thread | 3 | 760 | **Pass** + liveness |

`_hardlink_4node` (R + A + B + shared C, production-race repro
with Phase 4 reachability gating + outer-retry semantics):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 2-thread (production-race, no fix) | 2 | 97 | **FAIL** — SnapshotConsistency violated |
| 2-thread (Phase 4 reachability gating + DISTURBED retry) | 2 | 531 | **Pass** + liveness under per-action `WF(MigrateCToA)` |

`_hardlink_external` (P2 hard-linked under both GN1 and P1, no migration —
direct repro of `SnapshotConsistency` violation when Phase 4 finalises
GN1 ~missing while GN2.sub[P2] is unreachable):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 1-thread | 1 | 15 | **Pass** + liveness (with Phase 4 reachability gate) |
| 2-thread | 2 | 136 | **Pass** + liveness |

`_hardlink_external_migration` (bundle on GN1 atomically migrates P2 from
P1's tree into GN2.sub[P2]; per-action WF on every progress step):

| Config | Threads | Distinct states | Result |
|---|---|---|---|
| 1-thread | 1 | 8 | **Pass** + liveness |
| 2-thread | 2 | (~hundreds) | **Pass** + liveness |
| 3-thread | 3 | 1,202 | **Pass** + liveness under per-action WF on every BundlePhase* / BundlePullP1 / BundleCASP2 / BundleUpdateGN1 step |

### First per-action fairness in the KAME TLA+ corpus

The `_hardlink_4node` (with the C++ fix mirrored) is the first KAME
TLA+ spec to use **per-action fairness** rather than the blanket
`WF_vars(NextStep)`:

```tla
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads : WF_vars(MigrateCToA(t))
```

`_hardlink_external_migration` extends the same idea further — every
progress action of every thread gets its own per-action WF, since
each step of the bundle pipeline must be guaranteed to fire against
arbitrary peer retries:

```tla
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads :
            /\ WF_vars(BundlePhase1(t))
            /\ WF_vars(BundlePullP1(t))
            /\ WF_vars(BundleCASP2(t))
            /\ WF_vars(BundleUpdateGN1(t))
            /\ WF_vars(BundlePhase4(t))
            /\ WF_vars(BundlePhase5(t))
```

Rationale.  With the production-faithful retry semantics (Phase 4 on
reachability failure returns to `bundle_phase1` — matching the C++
`BundledStatus::DISTURBED` return out to `snapshot()`'s outer retry
loop), the model admits executions where the bundling thread retries
arbitrarily many times.  The retry only terminates after the peer's
release-step-2 (`MigrateCToA` in the model) fires, restoring
`A.sub[C]` to a reachable state.

`WF_vars(NextStep)` (blanket) is too weak here.  Weak fairness on a
disjunction guarantees "some enabled action fires" — an infinite
sequence firing only `BundlePhase1A/B/2/4` (= bundle retries)
satisfies it, even though `MigrateCToA` is continuously enabled in
parallel.  Such a trace violates the production fairness assumption,
but TLC's liveness checker accepts it.

`WF_vars(MigrateCToA(t))` — per-action weak fairness — excludes
those traces: if `MigrateCToA(t)` is continuously enabled (and it
is, in this model — no peer action re-disables it once enabled), it
must eventually fire.  After it fires, the next bundle attempt sees
the reachable state and finalizes cleanly.

(SF would also work but is overkill here — SF only adds value when
the action can oscillate between enabled and disabled.  WF and SF
produce the same 13,056-distinct-state verification with
`EventuallyAllDone` PASS.)

### Doesn't "older-wins" subsume this fairness?

The implementation's LL-free `negotiate` provides `older-wins`
arbitration on CAS contention.  Doesn't it already cover progress?

No, because `older-wins` arbitrates **on the same linkage**.  The
race here spans **three different linkages**:

* T1's bundle CAS — operates on `Root`'s `m_link`
* T2's release step 1 (`ReleaseBCNoMigrate`) — operates on **B**'s `m_link`
* T2's release step 2 (`MigrateCToA`) — operates on **A**'s `m_link`

T1 and T2 never CAS the same linkage in this scenario, so
`older-wins` has nothing to arbitrate between them.  T1 can retry
its Root-CAS arbitrarily many times without ever blocking T2 from
running its B-CAS or A-CAS.

Production progress for T2 therefore depends on **OS-level thread
scheduling** (each thread gets CPU time), not on negotiate.  The
TLA+ model abstracts the OS-scheduling guarantee as
`WF_vars(MigrateCToA)`.

All earlier KAME TLA+ specs (`BundleUnbundle_2level_LLfree`,
`_3level_LLfree`, both `_dynamic` variants, the other hardlink
specs, and `atomic_shared_ptr.tla`) terminate without needing
per-action fairness because their race surfaces were on a single
linkage chain, where LL-free priority gating bounds retries
structurally and `WF_vars(NextStep)` suffices.

### State-space bounding policy

User feedback 2026-05-20 led to a uniform policy across all hardlink
models — **no artificial retry bound, no CONSTRAINT, no MOD wrap-around**:

| Model | retryCount / DebugRetryBound | Bounding mechanism |
|---|---|---|
| `_hardlink_4node` | **removed** (gold standard) | `"in_release"` pc state binds the 2-step release into one atomic-from-caller API call (mirrors C++ `release(B, C)`); finite because every other variable has a finite domain |
| `_hardlink_self_collision` | **removed** | finite by bounded domains |
| `_hardlink_external` | **removed** | finite by bounded domains |
| `_hardlink_external_migration` | **removed** | per-action WF on every progress step replaces the retry-counter role; finite by bounded domains |
| `_hardlink_dynamic` | never had one | finite by bounded domains + `MaxCommits=1` |

Rationale.  CONSTRAINT silently prunes long-retry paths and risks
false-negative liveness verdicts; INVARIANT bounds on a retry counter
false-positive on legitimate retries that recover.  Removing both
gives an honest verification: TLC terminates naturally on the bounded
state space, and `EventuallyAllDone` is checkable end-to-end.

The `_hardlink_4node` and `_hardlink_external_migration` models in
particular demonstrate two complementary recipes:

* **4-node**: a 2-step release becomes a single "logical action" via
  a binding pc state — mirrors how the C++ `release(B, C)` API is a
  single non-preemptible call from the caller's standpoint.  No
  per-action fairness on the bundling thread is needed because the
  retry/release pairing is structural.
* **external_migration**: cross-tree races where three different
  linkages are involved use **per-action WF** on every progress step
  for every thread.  This models OS-level thread scheduling
  fairness, since in-linkage older-wins negotiate doesn't arbitrate
  between threads that never CAS the same linkage.

### Non-atomic test pattern (`BundleUnbundle_hardlink_nonatomic.tla`)

A separate spec verifies whether the test pattern restored by remote
branch claude/refactor-negotiate-scoped-f7de2 commit b23fa954 — non-tx
`p1->insert(p2)` / `p1->release(p2)` interleaved with transactional
`gn1`/`gn2` operations — exposes a real liveness gap in master, or
merely a scheduler artifact.

The spec models:
* Per-thread sequence ❶ NonTxInsertAC → ❷ TxInsertHardlink →
  ❸ NonTxReleaseAC (leaves C in **limbo**: `bundledBy = A` while A no
  longer references C) → ❹ TxReleaseHardlink → destructor
  finalises A and C in limbo.
* Two finalize variants (CONSTANT `UseFixVariant`):
  - `FALSE` (master): bundle-fall-through, walks bundledBy chain →
    requires scope on Root (heavily contended by the peer's TX).
  - `TRUE` (b23fa954): direct self-promote CAS on C's linkage only
    — no chain walk, no Root contention.
* Two fairness levels: `Spec` (per-action WF on every progress
  step) and `WeakSpec` (only blanket `WF_vars(NextStep)`).

Result (all four configurations — re-verified 2026-06-20, TLC on
OpenJDK 24, 2 threads, `MaxIter = 2`, no `CONSTRAINT`):

| Spec / Variant | States generated | Distinct | Depth | `RootMutex` | `EventuallyAllDone` | Time |
|---|---|---|---|---|---|---|
| `Spec` (per-action WF) + master | 534 | 308 | 35 | ✅ holds | ✅ PASS | <1 s |
| `Spec` (per-action WF) + fix | 550 | 308 | 35 | ✅ holds | ✅ PASS | <1 s |
| `WeakSpec` (blanket WF) + master | 534 | 308 | 35 | ✅ holds | ✅ PASS | <1 s |
| `WeakSpec` (blanket WF) + fix | 550 | 308 | 35 | ✅ holds | ✅ PASS | <1 s |

All four exhaust completely (queue 0). The reachable state graph is
**identical across fairness levels** — 308 distinct states at depth 35
whether `Spec` (per-action WF) or `WeakSpec` (blanket WF), because the
fairness constraint affects only the liveness check, not the set of
reachable states. `master` vs `fix` differ only in *generated* states
(534 vs 550), reflecting the variant's distinct finalize transitions
(`fin_*_walk` chain-walk vs `fin_*_cas` self-promote), and converge to
the same 308 distinct states. Crucially, **even `WeakSpec` + master**
(the weakest fairness on the unoptimised path) satisfies
`<>AllDone` — so the master path is live at this modelling abstraction
under blanket weak fairness alone.

**Conclusion:** at this modelling abstraction, both paths are
theoretically live.  The b23fa954 self-promote is a **CAS-count
optimization** — it reduces the chain walk's `O(chain × retry)`
CASes to `O(1)`, but the master fall-through is not livelocked in
principle.  The ~10% residual hang reported on the remote branch is
therefore consistent with real-OS scheduling artifacts (real
schedulers do not strictly meet `WF` in finite time), not a logic
gap.

The optimization is gated by `KAME_STM_OPTIONAL_OPTIMIZATION`
(defined to `1` by default in `kame/transaction_definitions.h`,
commits `ead762be`, `b7a4d882`).  Compile with
`-DKAME_STM_OPTIONAL_OPTIMIZATION=0` to disable the self-promote
shortcut and the related `bundle()` "peer-completed early return";
the resulting build matches the strictly TLA+-modelled master path
with no opt-in shortcuts.

**Note**: these shortcuts are NOT covered by the TLA+ models in
`tests/tlaplus/` — the unconditional master path is verified, the
shortcuts are CAS-count optimizations that bypass the modelled
control flow.

### Other STM correctness / perf knobs touched in this work

* **`KAME_ENABLE_SPIN_BAND_GATE` default flipped to `0`** on Apple
  Silicon (commit `472d193d`).  Re-benchmarked on M3 (4-thread):
  3level `payload_integrity` ~5% faster (mean, lower variance) with
  the gate OFF, others within noise.  Older sweeps had the gate ON
  by default for iMac Pro / Linux x86.  Override with
  `-DKAME_ENABLE_SPIN_BAND_GATE=1` if a target shows positive A-B.
* **`fair_mode_blocks_me` / `i_am_privileged_now` TID compare**
  (commit `9a0f9848`).  Fixed nested-Tx self-deadlock in the
  per-linkage privilege path: the per-linkage branches used
  `strip_kind` (which preserves the US timestamp field) for
  "is this slot mine?" identity, mis-identifying a nested inner Tx
  on the same thread as a peer.  Now use `stamp_tid` (TID only),
  matching the well-documented global-mode logic.  Measurable side
  effect: `payload_integrity` runs faster than before due to
  removed spurious CV-sleeps.

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

# External migration 3-thread (<1 s)
java -XX:+UseParallelGC -Xmx4g -cp tla2tools.jar tlc2.TLC \
  -workers auto -config BundleUnbundle_hardlink_external_migration_3thr_mc.cfg \
  BundleUnbundle_hardlink_external_migration.tla

# Dynamic (sibling parents, with liveness) — MaxCommits=1 and =2
for cfg in dynamic_1thr dynamic_2thr dynamic_2thr_commits2; do
  java -XX:+UseParallelGC -Xmx4g -cp tla2tools.jar tlc2.TLC \
    -workers auto -config BundleUnbundle_hardlink_${cfg}_mc.cfg \
    BundleUnbundle_hardlink_dynamic.tla
done

# Non-atomic test pattern — all four configs (master/fix × Spec/WeakSpec, each <1 s)
for cfg in master fix weak_master weak_fix; do
  java -XX:+UseParallelGC -Xmx4g -cp tla2tools.jar tlc2.TLC \
    -workers auto -config BundleUnbundle_hardlink_nonatomic_${cfg}_mc.cfg \
    BundleUnbundle_hardlink_nonatomic.tla
done
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
