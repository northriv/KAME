# Orphan-chain integration plan (atomic_shared_ptr refcount chunk-release)

Staged plan to replace the §36 orphan Treiber stack + `BIT_OWNED`
arbitration with the **TLA+-verified atomic_shared_ptr-refcounted intrusive
singly-linked orphan chain** (`tests/tlaplus/OrphanChain_atomicshared.tla`).

Branch: continuation of `orphan-atomicshared` (which already provides the
`force_intrusive_ref` / `atomic_intrusive_dispose` primitives in
`atomic_smart_ptr.h` and the `atomic_intrusive_dll_test.cpp` PoC).

## Design recap (what the model proved)

- **Self-ref on `m_filled`** (= `MASK_CNT`): a chunk holds ONE refcount to
  itself while `MASK_CNT > 0`, bumped at the `0->1` edge, dropped at the
  `1->0` edge (`atomicDecAndTest`-returns-true).  Decouples chunk lifetime
  from chain membership.  **One bump per non-empty/empty edge, NOT per slot**
  — the hot path (`atomicInc`/`atomicDecAndTest`) gains only a boundary
  branch, not an extra atomic.
- **Release = refcount -> 0** (the atomic_shared_ptr deleter), gated:
  `released  <=>  StructRefs = 0  <=>  no chain ref AND no pin AND MASK_CNT==0`.
- **Multiple sweepers, safe-side only**; **revival vs release** mutually
  exclusive via try_promote (`StructRefs>0`) vs release (`StructRefs=0`).
- TLC: `selfref` CLEAN (269), `noselfref` VIOLATION depth 3 (self-ref is
  load-bearing), `live` no-leak, `push` (head insert) CLEAN (541).

## Lifetime mapping: TLA+ action -> code site (from the integration map)

| TLA+ action | Current code site | Becomes |
|---|---|---|
| self-ref bump (`filled 0->1`) | `atomicInc(&m_flags_packed)` allocate_pooled FS=false (alloc.cpp:2070); FS=true alloc inc | bump chunk refcnt iff `MASK_CNT` was 0 |
| `Free` / self-ref drop (`1->0`) | `atomicDecAndTest(&c->m_flags_packed)` deallocate_pooled (alloc.cpp:1606,1639) | on true: empty-eager madvise (RSS) + drop self-ref |
| `Push` (owner-exit non-empty) | `orphan_push(c)` (alloc.cpp:3048) in `release_dll_chunks_for_thread` | atomic_shared_ptr Treiber push onto `g_orphan_head` |
| `Revive` (reuse adopt) | `orphan_pop()` + `BIT_OWNED` claim loop (alloc.cpp:2700-2756) | `try_promote` adopt from chain (acquire-if-nonzero) |
| `SweepRelink`/`SelfResetNext`/`HeadAdvance` | (none — stack has no compaction) | new `orphan_scrub()` GC pass (CAS relink past empties) |
| `Release` (refcount 0) | `deallocate_chunk` (alloc.cpp:3054) via `bucket_release_chunk` (3733) | the intrusive `atomic_intrusive_dispose` -> `deallocate_chunk` |
| owner-exit empty release | `release_dll_chunks_for_thread` empty branch (3010-3031) | drop self-ref -> refcount path |

## Chunk-header budget

64-byte header: `[0..39]` used (size_info / palloc / fn / sizeof_fn /
dedicated_size), **`[40..63]` = 24 free bytes**.  The embedded
`PoolAllocator` (chunk_base+64) already holds `m_flags_packed`,
`m_owner_id`, `m_dll_prev/next`.  The intrusive node needs: `atomic<Refcnt>
refcnt` (8B) + `atomic_shared_ptr<...> m_orphan_next` (8B).  Plan: place
`refcnt` + `m_orphan_next` in the embedded object (NOT the 24 header bytes —
those stay for `lookup_chunk`'s plain reads).  `m_orphan_next` REPLACES the
stack's reuse of `m_dll_next` (the two must not alias; orphan chain link is
distinct from per-thread DLL link).

## Stages (each gated by: build + `alloc_stress` residual=0 + `bench_loop` 64B non-regression)

All work behind `#if KAME_ORPHAN_CHAIN` (default **0**), so every stage
leaves the shipping path (orphan stack) intact and the build green until the
final flip.

- **Stage 1 — intrusive-node embed (scaffolding, flag OFF, unused).**
  `force_intrusive_ref<PoolAllocator<...>>`; add `Refcnt`/`refcnt` +
  `m_orphan_next` (atomic_shared_ptr) + `atomic_intrusive_dispose` ->
  `deallocate_chunk`.  Static-asserts mirror the PoC.  *Gate:* build
  identical, flag-OFF object code unchanged.

- **Stage 2 — self-ref accounting.**  Wire the `MASK_CNT` `0->1` / `1->0`
  edges to `refcnt` bump/drop (boundary-only).  *Gate:* flag-ON unit test
  asserts `refcnt == (MASK_CNT>0) + chain-refs`; hot-path disasm shows only
  a boundary branch added.

- **Stage 3 — owner-exit publish.**  Non-empty branch of
  `release_dll_chunks_for_thread`: push to `g_orphan_head` chain (Treiber,
  atomic_shared_ptr) instead of `orphan_push`.  Empty branch: drop self-ref
  (refcount path) instead of direct `deallocate_chunk`.  *Gate:*
  alloc_thread_churn scenario 2.

- **Stage 4 — empty-eager release.**  Cross-free `atomicDecAndTest`-true:
  `madvise` slot pages immediately (RSS, chain-independent) + drop self-ref;
  chunk released when refcount hits 0.  *Gate:* reserved plateau (macOS
  too — re-check the churn gate `tests/CMakeLists.txt`).

- **Stage 5 — reuse adopt.**  Replace `orphan_pop` + `BIT_OWNED` claim with
  `try_promote` adopt from the chain (acquire-if-nonzero = the revival
  gate).  Non-empty orphans adopted whole; the per-thread DLL re-splice as
  today.  *Gate:* alloc_stress + bench.

- **Stage 6 — sweep compaction.**  `orphan_scrub()`: walk the chain holding
  `local_shared_ptr` pins, CAS `pred->m_orphan_next` past empty nodes,
  idempotent-null a dead node's own next.  Multi-sweeper (no
  serialization).  Trigger at owner-exit (garbage-aligned) + opportunistic
  at adopt.  *Gate:* alloc_thread_churn + a churn stress with K survivors.

- **Stage 7 — retire + flip.**  Remove `s_orphan_head`/`orphan_push`/
  `orphan_pop`, the `BIT_OWNED`-arbitration release paths subsumed by
  refcount, and any now-dead seqlock.  Default `KAME_ORPHAN_CHAIN = 1`.
  *Gate:* full `ctest`, `alloc_stress` residual=0, `bench_compare.sh`
  non-regression (esp. 64B 1T/MT), `alloc_thread_churn` on macOS+Linux,
  STM `3level_mixed`.

## Invariants carried from the model (assert in debug builds)

1. `released(c) => MASK_CNT(c)==0` — the deleter's DEBUG_GUARD (the
   `Inv_NoBadRelease` the model enforces).
2. `MASK_CNT(c)>0 => refcnt(c)>=1` — self-ref (lifetime decoupling).
3. a live chunk's `m_orphan_next` never points at a released chunk
   (`Inv_NoDanglingNext`).
4. adopt (`Revive`) only via successful `try_promote` (refcnt was >0).
5. chain stays acyclic; relinks reachability-preserving + dead-only.

## Risks / watch-items

- **Hot path**: Stage 2's boundary branch must NOT add an atomic to the
  per-slot inc/dec.  Verify by disasm (objdump) + 64B bench.
- **`m_orphan_next` vs `m_dll_next`**: distinct fields — the orphan chain
  and the per-thread DLL must not alias links (the §36 stack reused
  `m_dll_next`; the chain cannot, since an orphan may be both chain-linked
  and transiently pin-walked).
- **Disposer re-entrancy**: `atomic_intrusive_dispose -> deallocate_chunk`
  must not allocate (it doesn't) and must run the madvise/claim-clear once.
- **LOCAL_REF_CAPACITY contention** on the hot `g_orphan_head` under many
  concurrent adopters — measure.
