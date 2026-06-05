# §36b Orphan-reuse array — HAND-OFF (needs TLA before completion)

**Status: INCOMPLETE / UNSAFE. Do NOT merge the WIP code.** The current
production code (`master` / §36, Treiber stack + 18-bit ABA tag) is
**corruption-free but LEAKS** (see below). The §36b replacement designed and
prototyped here fixes the leak but still has **≥2 unresolved concurrency
races** under heavy stress; it must be modelled in TLA+ before it can be
trusted, especially on macOS (which the AI agent that wrote this could not
run).

The WIP patch lives in a `git stash` on branch `claude/youthful-mayer-FUhq2`
(see "Retrieving the WIP" below).

---

## 1. The problem being solved

§36 (`dcd3df7`, on `master`) reuses orphaned chunks via a **per-template
lock-free Treiber stack** (`s_orphan_head`). A chunk left non-empty by an
exited owner is `orphan_push`ed; an allocating thread `orphan_pop`s it before
mmap'ing fresh. The stack is the *only* drain path.

**Leak:** a chunk that *empties* while on the stack is **never released** —
§36 removed the cross-thread dec-to-0 release precisely so the stack node stays
mapped for `orphan_pop`'s `m_dll_next` deref. If a template's stack is never
popped (its size class is not allocated again — e.g. **per-thread
infrastructure chunks** of a size the workload never re-allocates), its empty
chunks accumulate **without bound**.

* `tests/alloc_thread_churn_test.cpp` reproduces it. On **macOS** both
  scenarios ramp linearly (~+2 chunks per thread spawn/exit, pattern-
  independent → it is the **per-thread teardown**, not the survivor logic).
  On **Linux** it is flat (TLV vs pthread-key teardown order differs: Linux
  releases those infra chunks empty instead of `orphan_push`ing them).
* Baseline §36 stress: **0 corruption** (`alloc_stress` 0/120) — the 18-bit
  Treiber tag makes it ABA-safe. It just leaks.

**Goal of §36b:** reuse (alloc path) **and** release-on-empty (dealloc path)
**and** ABA-safe, **bounded** (no leak at any K).

---

## 2. §36b design (prototyped in the stash)

Replace the Treiber stack with a **bounded, versioned array**:

* `s_orphan_slots[K]`, `K = 32`, **per (ALIGN, FS) template** (FS=false: the
  buckets sharing an ALIGN share one array — 4 arrays for 32/64/256/1024;
  FS=true: one per size). A static member, like the old `s_orphan_head`.
* Each slot is a **versioned tagged word** (`std::atomic<uint64_t>`): low 48
  bits = chunk pointer (`< 2^48`, guaranteed by the §35 `RADIX_VA_LIMIT` mmap
  gate), high 16 bits = an **ABA version** bumped on **every** push / pop /
  release-take. Empty == low-48 zero (version kept live across empty).
* Per-chunk **disposition arbiter** `m_orphan_disp` (one atomic byte, declared
  next to `m_dll_prev/next` for layout-stable `<ALIGN,DUMMY,DUMMY>` access):
  `ORPHAN_OWNED` (off-array) / `ORPHAN_PUSHING` / `ORPHAN_RELEASED` /
  `1+slot` (on the array at that slot).
* **Dispersed scan start** `kame_owner_id() & (K-1)` (LRC-style) for push/pop.

**Ownership invariant (the intent):** the relevant CAS winner owns the chunk's
fate.
* *Not-yet-listed* (OWNED): `m_orphan_disp` CAS — `orphan_push` (OWNED→PUSHING,
  lists for reuse) vs free-on-empty (OWNED→RELEASED, releases). Winner owns.
* *On the array* (1+slot): the **versioned slot CAS** — `orphan_pop` (reuse)
  vs free-on-empty (release). Winner owns; the version rejects a same-slot
  ABA re-push.

Bounded at K ⇒ **leak-free at any K**: empties are released directly (or sit
re-ownable on the array, ≤ K), overflow non-empty orphans are not listed but
are released the moment their last survivor is freed.

---

## 3. Implementation map (what changed, where)

`kamepoolalloc/allocator_prv.h`
* Replaced the `s_orphan_head` decl block with: `ORPHAN_K/OWNED/PUSHING/
  RELEASED`, `ORPHAN_PTR_MASK/VER_INC`, `s_orphan_slots[ORPHAN_K]`
  (`std::atomic<uint64_t>`), and decls `orphan_push`, `orphan_pop`,
  `orphan_claim_for_release` (bool), `orphan_release_self`.
* Added `std::atomic<uint8_t> m_orphan_disp{ORPHAN_OWNED};` **next to
  `m_dll_prev/next`** (layout-stability requirement — do NOT move it to the
  static block).

`kamepoolalloc/allocator.cpp`
* `s_orphan_slots` definition; `orphan_push` / `orphan_pop` /
  `orphan_claim_for_release` (versioned) / `orphan_release_self`
  (destructor + `deallocate_chunk`, with a **DEBUG abort guard** that fires on
  `ORPHAN_RELEASE_BAD` = releasing a non-empty/owned chunk).
* `batch_return_to_bitmap` **FS=true** and **FS=false**: the OnClearFn now,
  on `atomicDecAndTest()==true`, calls `release_me =
  this->orphan_claim_for_release()` (atomic claim at the dec-to-0 instant),
  and the **destructive** `orphan_release_self()` is run **after
  `batch_clear_impl` returns** (it touches `this` post-loop).
* `orphan_pop` caller (`allocate_chunk_path`, ~line 2726): after re-owning,
  `oc->m_orphan_disp.store(ORPHAN_OWNED, relaxed)`.
* `release_dll_chunks_for_thread` non-empty branch still calls `orphan_push(c)`
  (signature unchanged); empty branch still releases directly.

---

## 4. Races found and handled (during prototyping)

1. **UAF after release inside OnClearFn.** `batch_clear_impl` touches `this`
   *after* the per-word loop (`m_last_coalesce_x16` for FS=true; the max-n-gate
   `m_flags_filled_cnt` for FS=false). Releasing `this` inside the OnClearFn
   is a use-after-free. → **Deferred** the destructive release to after
   `batch_clear_impl` returns.
2. **Re-own during the deferral.** Deferring the *whole* release lets an
   `orphan_pop` re-own + reuse the chunk in the gap, after which the deferred
   release frees a **live** chunk. → Split: **`orphan_claim_for_release()`**
   (atomic ownership CAS, runs *inside* the OnClearFn at dec-to-0) +
   **`orphan_release_self()`** (destructive, deferred). Only the destructive
   half is deferred.
3. **Same-slot ABA.** A free-on-empty that read `disp==1+slot` could ABA-match
   a chunk meanwhile popped → re-owned → reused → re-pushed into the **same
   slot**, releasing a live chunk. → **Per-slot 16-bit version**; the take-CAS
   expects the exact `(ptr,ver)` word.

---

## 5. OPEN — unresolved races (this is why TLA is needed)

With all three fixes above, `alloc_stress` (2000 threads, concurrent=32,
cross_pct=10%, K=32) still fails at **~1.7 % per run** with **two** modes:

* **`ORPHAN_RELEASE_BAD` (ABA) — REDUCED by versioning but NOT eliminated.**
  e.g. run 116/300: `flags=8000_0001` (BIT_OWNED set + MASK_CNT 1) at
  `orphan_release_self` ⇒ released a **re-owned, live** chunk. So the version
  does not fully close the pop / release / push / re-own arbitration.
* **`sentinel mismatch` (double-allocation).** e.g. run 7/300: a slot's user
  bytes were overwritten ⇒ the same memory was handed to two allocations.
  Likely a chunk released (units → region bitmap) and re-claimed while a live
  reference remained, or orphan-reuse double-counting.

Baseline §36 was **0/120** clean, so these are **introduced by §36b** (or its
timing exposes a latent interaction).

### Prime suspect for TLA to examine

The **`orphan_pop` caller re-owns in 3 non-atomic steps**:
`orphan_pop` (versioned slot-take) → claim `BIT_OWNED` (CAS preserving
`MASK_CNT`) → re-arm (`m_owner_id`, **`m_orphan_disp = OWNED`**, DLL splice).
During this window the chunk is **off the array, `disp` still `1+k` (stale),
`BIT_OWNED` clear, `MASK_CNT` = survivors**. A concurrent cross-thread free of
the last survivor hits `atomicDecAndTest()==true` → `orphan_claim_for_release`
runs against the **stale `disp`**. The versioned slot CAS *should* reject it
(slot already taken), but the interplay of:
* the caller's `BIT_OWNED`-claim CAS preserving a `MASK_CNT` that a concurrent
  free is decrementing, and
* the timing of the `disp = OWNED` reset vs the slot-version lifecycle,

has at least one reachable interleaving that still releases or double-allocates.
Model these as separate atomic steps.

---

## 6. Verification status

| | result |
|---|---|
| Linux churn (both scenarios) | **leak FIXED — plateau** (reserved 96→96 MiB, chunks_live 37/103 flat) |
| Linux `alloc_bucket34_repro` | 0/30 |
| Linux ctest suite (c_api/huge/stats/intercept/evict/realtime/ros2/pmr/aligned) | all green |
| Linux `alloc_stress` (2000 thr, 21.3M ops) | **~1.7 % corruption** (ABA + sentinel) ← BLOCKER |
| macOS | **not run** (agent limitation) |
| Baseline `master` §36 stress | 0/120 (corruption-free; **leaks**) |

---

## 7. Retrieving the WIP

```sh
git stash list          # stash@{0}: "WIP §36b versioned-slot orphan array — ..."
git stash apply         # restores the 2-file diff (allocator.cpp, allocator_prv.h) onto master(§36)
# reproduce:
cmake -S kamepoolalloc/tests -B build && cmake --build build
for i in $(seq 1 300); do ./build/alloc_stress_test || echo "fail $i"; done   # ~1.7% ORPHAN_RELEASE_BAD / sentinel
./build/alloc_thread_churn_test                                               # leak FIXED (plateau)
```

The `ORPHAN_RELEASE_BAD` DEBUG guard in `orphan_release_self()` aborts loudly
on the ABA mode — keep it during bring-up; it does not catch the sentinel mode.

NOTE: the stash is local to the dev container and is **not durable**. If this
container is recycled the diff is lost — re-apply or commit it somewhere before
relying on it.

---

## 8. Recommendations

1. **Keep `master` §36** (leaks, but **no memory corruption**) until §36b is
   **TLA-verified**. There is no urgent corruption bug on `master`.
2. **TLA model** the §36b state space: per-chunk `(m_orphan_disp, BIT_OWNED,
   MASK_CNT)` × per-slot `(ptr, version)` × the operations
   {`orphan_push`, `orphan_pop`+claim+re-arm, free→`orphan_claim_for_release`,
   `orphan_release_self`}. Invariants to check:
   * a chunk is reachable from **at most one** of {a slot, one thread's DLL,
     released};
   * a chunk taken via `orphan_release_self` has `MASK_CNT==0` **and** is in no
     slot **and** no thread is mid-re-own;
   * no slot / region unit is handed to two live allocations.
   Treat the `orphan_pop` caller's re-own as **three separate atomic steps**.
3. **Cheaper alternative if reuse is not essential ("option C"):** drop the
   array and simply **restore release-on-empty** for the cross-thread dec-to-0
   (the pre-§36 behaviour). This is **provably simple, corruption-free, and
   leak-free** — it was prototyped earlier (feature-branch commit `1e65bdd`,
   "option B"): Linux churn plateaus on both scenarios with zero stress
   corruption. It only loses the reuse of *non-empty* orphans' free slots
   (they drain + recycle warm instead of being re-owned in place).

---

*Generated during an interactive design session; see the conversation for the
full derivation. The numbers above are Linux (x86-64, 4 KiB pages). macOS
(16 KiB pages, MADV_FREE, TLV teardown order) must be validated separately.*
