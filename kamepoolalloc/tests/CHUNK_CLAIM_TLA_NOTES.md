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

---

## 12. Follow-up "(1b)": chunk-header-skip + local-index freelist + cache-line-align

**Status: design fixed (cache-line-aware), implementation pending.**
Builds on "(1)" (f3d90835, chunk-local freelist, +24% global-dynamic).

### Two coupled optimisations

**(2) skip chunk_header in the dealloc fast path.**  The embed layout
makes `chunk_obj = chunk_base + ALLOC_CHUNK_HEADER` a constant, so the
fast path need NOT read `palloc` (chunk_header[8]) at all — derive the
object address arithmetically.  But `palloc` skip alone is useless: the
bucket is currently derived from `size_info` (chunk_header[0], SAME
cache line 0), so the line is still touched.  To truly drop cache
line 0, the bucket must come from elsewhere (see (1)).

**(1) local-index freelist + bucket info in chunk_obj.**  Move the
bucket determinant off chunk_header onto chunk_obj:
  - `m_base_bucket` (chunk's lowest bucket).  FS=true: that's the one
    bucket; FS=false: the base for local indexing.
  - bucket at dealloc: FS=true -> m_base_bucket (no size_info read);
    FS=false -> slot prefix at `p-8` (slot-adjacent, NOT chunk_header).
  - `m_freelist_head[]` indexed by `bucket - m_base_bucket` (LOCAL),
    so an FS=false chunk's few SIZEs pack into ONE cache line instead
    of being sparse across the global-48 layout.
  - released signal moves from `palloc==0` to `m_owner_id==0` (cleared
    by deallocate_chunk); the owner-id check already gates the fast
    path, so a released/foreign chunk (owner_id 0 or mismatched) falls
    to the slow path where palloc is read for the released test.

### Resulting fast path (touches cache line 0 = chunk_header NEVER)

    s_back_offset[unit] -> chunk_base -> chunk_obj = chunk_base+HEADER
    if chunk_obj->m_owner_id == s_tls_owner_id:        // chunk_obj line
        bucket = FS ? *(p-8)prefix : chunk_obj->m_base_bucket
        chunk_obj->m_freelist_head[bucket - m_base_bucket].push(p)

Lines touched: s_back_offset + chunk_obj (its own line) + (FS=false:
slot prefix near p).  chunk_header (palloc/size_info) is NOT read.

### CACHE-LINE-AWARENESS (critical — line size is arch-dependent)

chunk_obj = chunk_base + 64.  On a 64 B line, chunk_header (0..63) and
chunk_obj (64..) are different lines already.  On a **128 B line
(Apple Silicon aarch64)**, chunk_header (0..63) and chunk_obj (64..127)
share line 0 — so the skip buys nothing unless chunk_obj's hot members
are pushed to the NEXT line.

Use `KAME_CACHE_LINE` (kamestm/transaction_detail.h: Apple-aarch64 128,
PPC 128, Fujitsu-aarch64 256, else 64; copy the macro into
kamepoolalloc for standalone independence):

    struct PoolAllocatorBase {
        // ... (vtable, etc., or keep chunk_header fully OUTSIDE the
        //      object in chunk_base[0..63] as today)
        alignas(KAME_CACHE_LINE) uint32_t m_owner_id;   // forced onto
        uint16_t m_base_bucket;                          //   a fresh
        char *m_freelist_head[8];                        //   cache line
        // ...
    };

chunk_base is 256 KiB-aligned, so `alignas(KAME_CACHE_LINE)` lands the
hot block on chunk_base + KAME_CACHE_LINE (128 on Apple), guaranteeing
a line distinct from chunk_header on every target.  Cost: up to
KAME_CACHE_LINE-64 B of pad inside the chunk (negligible vs 256 KiB).

### freelist_head[8] sizing

FS=false buckets 24..47 split across 4 ALIGN templates (~6-8 each);
[8] covers the widest.  static_assert each FS=false template's
bucket-span <= 8 at instantiation.  FS=true uses index 0 only.

### Implementation order

1. copy KAME_CACHE_LINE macro into allocator_prv.h.
2. PoolAllocatorBase: `alignas(KAME_CACHE_LINE)` hot block
   {m_owner_id, m_base_bucket, m_freelist_head[8]}; drop the global
   m_freelist_head[48].
3. ctor: set m_base_bucket (template-derived), zero the 8 heads.
4. deallocate fast path: chunk_obj = chunk_base+HEADER (no palloc
   read); owner_id check; bucket via base_bucket (FS=true) / p-8
   (FS=false); local-index push.  No chunk_header read.
5. new_redirected: chunk-local pop via local index (g_thread_chunks
   gives chunk; chunk->m_base_bucket gives the offset).
6. deallocate_chunk: m_owner_id = 0 (released signal); slow-path
   palloc==0 test stays for the cross/foreign cases.
7. drain: per-chunk, local-index walk (8 heads, not 48).
8. bench: global-dynamic (expect further gain from the dropped
   chunk_header line) AND Apple-aarch64 128 B (verify the alignas
   actually separates the lines — perf c2c / instruments).

All-or-nothing landing like (1); verify cross=100 + initial-exec +
global-dynamic + (ideally) an aarch64 run before commit.

### 12.1 The chunk_obj hot block = redundant copy of chunk_header's fast-path info

Refinement (per review): for the fast path to NEVER touch cache line 0,
copy EVERY chunk_header datum the fast path needs into the cache-line-1
hot block.  The fast path needs exactly two things from chunk_header:

  1. the FS=true/false discriminator (today: low 32 b of size_info);
  2. the bucket (FS=true: from size_info's ALIGN; FS=false: slot prefix).

So the alignas(KAME_CACHE_LINE) hot block carries BOTH as redundant
copies, valid for FS=true AND FS=false:

    alignas(KAME_CACHE_LINE) struct {
        uint32_t m_owner_id;     // owner check
        uint8_t  m_fs_flag;      // <- copy of size_info's FS discriminator
        uint16_t m_base_bucket;  // <- copy of the bucket (FS=true) / base
                                 //    bucket for local indexing (FS=false)
        char    *m_freelist_head[8];   // local-indexed
    };

Fast path then reads ONLY the hot block (+ the slot's own prefix at p-8
for FS=false, which lives on the slot's cache line, not chunk_header):

    chunk_obj = chunk_base + ALLOC_CHUNK_HEADER;     // no palloc read
    if (chunk_obj->m_owner_id == s_tls_owner_id) {   // cache line 1
        unsigned b = chunk_obj->m_fs_flag ? *(uint32_t*)(p-8)
                                          : chunk_obj->m_base_bucket;
        chunk_obj->m_freelist_head[b - chunk_obj->m_base_bucket].push(p);
        return true;
    }
    // slow path: read chunk_header (palloc released-check, vtable fn)

  FS=true  : cache line 1 only.
  FS=false : cache line 1 + slot prefix (slot's own line).
  Neither reads chunk_header's cache line 0.

These copies are written once at allocate_chunk (alongside m_owner_id)
and are immutable for the chunk's life, so no coherence cost beyond the
initial store.  size_info / palloc / fn stay in chunk_header for the
slow path (cross-thread / released / foreign) and for size_of's
SizeOfFn dispatch — only the OWNER-FREE fast path is moved off line 0.
