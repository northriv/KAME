# Chunk-Claim / back_offset / recycle protocol — formal-verification notes

Notes for TLA+ modelling of the kamepoolalloc **chunk-claim** layer
(distinct from the STM specs in `kamestm/tests/tlaplus/`, which model
`atomic_shared_ptr` / STM commit / bundle).  This layer is the
allocator's region → unit → chunk claim/recycle protocol.

A real lock-free data race in this protocol caused ~10–47 %
double-allocation failures in `alloc_stress_test`; the dominant path
was fixed ad-hoc (see commit "fix back_offset speculative-write data
race").  A complete root-cure is the goal of the formal model.

---

## 1. Shared global state (static in `PoolAllocatorBase`)

Shared across **every** `(ALIGN, FS)` template instantiation — this
sharing is the crux.

| Variable | Type | Meaning |
|---|---|---|
| `s_mmapped_spaces[R]` | ptr | region base (32 MiB). `R` ≤ `ALLOC_MAX_MMAP_ENTRIES` |
| `s_claim_bitmap[R][W]` | atomic word | **2 bits / unit**: bit `2u` = claim, bit `2u+1` = ready |
| `s_back_offset[R][unit]` | byte | distance from `unit` back to its chunk's base unit (0 = base, 1..3 = continuation) |
| `s_region_has_free[]` | atomic | per-region "has free space" skip bitmap |

Region = 128 units; unit = 256 KiB; region = 32 MiB.

## 2. The key structural subtlety

**The same region's `s_claim_bitmap` / `s_back_offset` are scanned by
multiple templates with DIFFERENT `CHUNK_UNITS` strides.**

- `CHUNK_UNITS` = 1 (`ALIGN<256`) / 2 (`ALIGN<1024`) / 4 (`ALIGN≥1024`)
- `CHUNK_STRIDE_BITS = 2·CHUNK_UNITS`
- The bug manifests **only** under cross-stride contention. A model
  needs **≥ 2 distinct `CHUNK_UNITS` procs** (e.g. 1-unit + 2-unit);
  a single-stride model cannot reproduce it.

## 3. Per-chunk local state

| Variable | Meaning |
|---|---|
| `m_flags[word]` | slot-occupancy bitmap (1 bit/slot FS=true; N bits/slot FS=false) |
| `m_flags_packed` | bit31 = `BIT_OWNED`; bits0..30 = `MASK_CNT` = **count of non-empty `m_flags` words** (NOT slot count) |
| per-thread `freelist[bucket]` | owner-free target. **bit stays SET**, `MASK_CNT` unchanged |
| per-thread cross-dealloc batch | non-owner frees, deferred; flush clears bitmap bits |

## 4. Actions / transitions

- **claim** (`try_claim_in_region`): scan for `CHUNK_OCC_MASK == 0` run
  → CAS the `CHUNK_UNITS` claim bits → **write `back_offset` (BUG: was
  before the CAS; FIX: after)** → `create()` → write header(palloc) →
  `fetch_or(ready, release)`.
- **alloc_slot** (`allocate_pooled`): CAS-set a free bit in `m_flags`
  → **`if(oldv==0) atomicInc(MASK_CNT)` — NOT atomic with the CAS**.
- **free_owner**: freelist push; bit stays set; `MASK_CNT` unchanged.
- **free_cross / drain** (`batch_clear_impl`): CAS-clear bits →
  **`if(word nonzero→0) atomicDecAndTest(MASK_CNT)`**; if it reaches 0
  with `BIT_OWNED==0`, caller is the unique releaser.
- **lookup_chunk(p)**: locate region → read `back_offset[unit]` →
  compute base → read `palloc` from header.
- **deallocate_chunk**: clear ready → clear header → clear
  `back_offset` (step 4) → clear claim bits (step 5 ⇒ recyclable).
- **owner_release**: observe `MASK_CNT==0` → `atomicFetchAnd(~BIT_OWNED)`
  → if result 0, release.
- **release_dll_chunks_for_thread**: at thread exit, release empty
  chunks / clear `BIT_OWNED` on non-empty.

## 5. Invariants to check

- **INV1 no-overlap**: no two live chunks own the same unit.
- **INV2 back_offset-consistency**: for every claimed unit `u`,
  `lookup_chunk(any addr in u)` == the chunk that claimed `u`.
  **← the invariant the fixed bug violated.**
- **INV3 no-recycle-while-live**: a chunk's claim bits are cleared
  (units recyclable) only when all its `m_flags == 0`.
- **INV4 MASK_CNT-accuracy**: at every release decision point,
  `MASK_CNT==0 ⟹ ∀word m_flags[word]==0`.
- **INV5 bit↔slot bijection**: a slot free clears exactly that slot's
  bits, in the correct chunk.

## 6. The fixed bug (should be reproducible in the pre-fix model)

`back_offset` was published **before** the claim CAS:

```cpp
for(u in 0..CHUNK_UNITS) s_back_offset[bo_base+u] = u;   // speculative
writeBarrier();
if(bm->compare_exchange_weak(v, newv, ...)) { ...create chunk... }
```

Cross-stride race:

```
T1 (2-unit) claims units [2,3]: back_offset[2]=0,[3]=1; CAS WINS.
T2 (4-unit) tries  units [0..3]: back_offset[2]=2,[3]=3 (CLOBBER);
                                 CAS FAILS, retries elsewhere.
=> back_offset[2] = 2 (T2 stale), should be 0 (T1).
```

`lookup_chunk(slot in unit 2)` → wrong base → wrong chunk → a
cross/drain free clears a bit in the WRONG chunk → recycles a live
slot → double-alloc.

A 2-proc, distinct-`CHUNK_UNITS`, overlapping-unit-range model should
yield an **INV2** counterexample on the pre-fix spec.

## 7. Residual race (root-cure target, ~0.24 %)

After the fix, an over-clear probe still fired ~1 / 470 instrumented
runs (no sentinel corruption in this workload). Hand analysis did not
close it. Prioritised candidates for the model:

1. **`MASK_CNT` inc/dec non-atomicity (most likely).**
   `alloc_slot`'s `atomicInc` is a separate instruction from the
   bit-set CAS. Across 3+ procs interleaving with `free_cross`'s
   `atomicDecAndTest`, can `MASK_CNT` desync from the real bits so
   that `owner_release` / the cross-releaser (`i_am_releaser` in
   `batch_clear`) releases while a bit is still set?
   - The `owner_release` path was probed empirically (did not fire).
   - **The cross-releaser path (`atomicDecAndTest→0`) was NOT probed —
     focus here.**
   - Candidate interleaving: two procs alloc into the same empty word
     (both read `oldv==0`) → only one `inc` → the other races a `dec`
     on a different word.

2. **`deallocate_chunk` step4 (back_offset clear) vs step5 (claim
   clear) non-atomicity** × a concurrent `lookup` read-during-write
   on a continuation unit.

3. **cross-batch holding a deferred `(chunk, slot)` across a chunk
   release** — the invariant "a set slot bit keeps its chunk alive
   until flush" must hold; it collapses simultaneously with INV4.

## 8. Memory ordering (for RC11 mapping)

- claim CAS: success = `acquire`, fail = `relaxed`.
- ready: `fetch_or(release)`.
- `back_offset`: plain store + `writeBarrier()` (release fence) before
  `create()`.
- `m_flags` CAS / `MASK_CNT` inc/dec: `__sync_*` (acq_rel / seq_cst).
- `lookup_chunk`: plain `back_offset` load (no acquire fence — also
  worth checking).

## 9. Reproduction (empirical)

- Config: `alloc_stress_test 10000 16 1000 20`
  (total / concurrent / ops-per-thread / cross-%). Pre-fix ~40 %,
  post-fix 0 / 500.
- Strongly correlated with **thread-teardown count** (`total_threads`).
- Detection: per-allocation sentinel paint + a temporary over-clear
  probe (`(oldv & mask) != mask` in `batch_clear_impl`).

## 10. Suggested minimal model

- 1 region / 4–8 units / 2 procs / `CHUNK_UNITS ∈ {1, 2}` /
  alloc-free-recycle over a few steps.
- Structural finiteness via **preconditions** (cap unit count), **not**
  `StateConstraint` (per the project's TLA+ convention — see
  `kamestm/tests/tlaplus`).
- Pre-fix spec → expect **INV2** violation. Post-fix spec → INV2
  holds; then hunt **INV4** for the §7 residual.

---

## 11. Follow-up "(1)": chunk-local freelist (dealloc 2→1 TLS read)

**Status: design fixed, implementation pending.** Current master
(d7179d65) is stable: pre-WIP-equal perf (initial-exec 198 M ops/s),
all race fixes in force.

### Problem

`deallocate`'s owner-free hot path does 2 TLS reads:
- `g_thread_chunks[bucket]` (owner check: `== palloc`)
- `g_thread_slots[bucket].push(p)` (freelist push)

These are two distinct TLS symbols → 2 `__tls_get_addr` calls under
global-dynamic. (Under initial-exec they are 2 plain offset loads, so
the win is small — the real perf restore was initial-exec eliminating
`__tls_get_addr`, not the read count.)

### Design

Move the per-bucket freelist head OUT of the TLS array INTO the chunk:

- `PoolAllocatorBase` gains:
  - `uint32_t m_owner_id;`  — owner-thread id (cache line 1+, i.e.
    chunk_base+64, separate from chunk_header.palloc at chunk_base+8
    cache line 0, so owner's frequent freelist writes don't
    false-share the cross-thread-read palloc).
  - `char *m_freelist_head[ALLOC_NUM_BUCKETS];` — chunk-local
    per-bucket freelist heads. Global bucket index (sparse per chunk;
    FS=true uses 1 entry, FS=false uses the SIZEs it hands out).
    384 B/chunk — negligible vs 256 KiB chunk.
- New TLS: `s_tls_owner_id` (one id per thread, global counter).
- `deallocate` owner-free path: `palloc->m_owner_id == s_tls_owner_id`
  (TLS 1 read: s_tls_owner_id) → `palloc->m_freelist_head[bucket]`
  push (TLS 0 — chunk-relative). = **1 TLS read**.
- `new_redirected` alloc: `chunk = g_thread_chunks[bucket]` (TLS 1) →
  `chunk->m_freelist_head[bucket]` pop. = 1 TLS read (unchanged
  count, but freelist is now chunk-local).
- `drain_thread_slot_freelists` → walk each live chunk's
  `m_freelist_head[]` instead of g_thread_slots.
- Retire `g_thread_slots[]` (+ kame_slots_base, macOS TSD slot key).

### Risk

The freelist changes from **bucket-wide / chunk-spanning**
(g_thread_slots[b] held free slots from ANY chunk) to **chunk-local**
(only the active chunk's free slots for that bucket). This changes the
freelist hit-rate: a free slot in a non-active chunk is no longer
popped on the next alloc — it waits until that chunk is active again.
Must bench alloc_stress wall-clock + freelist-hit before/after; if
hit-rate drops materially, reconsider (e.g. keep g_thread_slots for
FS=false, chunk-local only for FS=true).

### Implementation order (9+ interdependent edits)

1. move/forward-declare ALLOC_NUM_BUCKETS before PoolAllocatorBase
   (or hardcode 48 with static_assert).
2. add m_owner_id + m_freelist_head[48] to PoolAllocatorBase.
3. s_tls_owner_id TLS + global-counter assignment.
4. PoolAllocator ctor: init m_owner_id, m_freelist_head[].
5. create(): embed offset += sizeof(new members) (~392 B); recompute
   slot-region start.
6. new_redirected: chunk-local freelist pop.
7. deallocate (FS=true/false + the 2418/965/1287 push sites):
   owner-id check + chunk-local push.
8. drain_thread_slot_freelists: per-chunk freelist drain.
9. retire g_thread_slots / kame_slots_base / macOS slot TSD.

Build breaks between edits 5–9 (hot path half-migrated), so this is an
all-or-nothing landing — do it in one focused pass, verify with
cross=100 (no double-payout) + alloc_minimal_bench (initial-exec) +
alloc_stress hit-rate, then commit.
