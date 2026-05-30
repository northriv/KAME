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

### 12.2 Implementation (LANDED) — two deviations from the §12/§12.1 sketch

Implemented on master.  Two corrections to the sketch were forced by the
actual bucket layout and the embed object's placement alignment:

**(a) Freelist stays GLOBAL-bucket-indexed [48], NOT local-index [8].**
The §12/§12.1 sketch assumed each FS=false chunk's buckets form a
contiguous span <= 8, so `bucket - m_base_bucket` would index a packed
[8] array.  That is FALSE.  An FS=false chunk is one `PoolAllocator
<ALIGN,false>` instantiation, and ALL same-ALIGN buckets share its one
`s_tls.my_chunk<ALIGN>` (see allocator.cpp §"Multiple bucket indices
share one PoolAllocator<ALIGN,false>"), so one physical chunk hands out
EVERY bucket of its ALIGN tier:

    ALIGN=32  -> buckets {6,8,10,12,14}      span 9   (6..14)
    ALIGN=64  -> buckets {24..31}            span 8
    ALIGN=256 -> buckets {16, 32..39}        span 24  (bucket 16, a LOW
                 FS=false bucket, shares the 256-B template with 32..39!)
    ALIGN=1024-> buckets {40..47}            span 8

`bucket - m_base_bucket` would reach 23 for the ALIGN=256 tier — far
past [8].  A per-tier remap (16->0, 32..39->1..8, etc.) differs per
ALIGN and would add a dependent load on the hot path, defeating the
cache-line win.  So the heads stay `m_freelist_head[ALLOC_NUM_BUCKETS]`
(hardcoded 48, static_assert'd) indexed by the global bucket, exactly as
follow-up "(1)" had them.  This costs 384 B/chunk (negligible vs 256 KiB)
and — crucially — the high-frequency small FS=true buckets (1..6) land at
hot-block offset 8..56, co-resident with m_owner_id on the SAME cache
line, so the common case is still one line.  The "(1b)" win is the
chunk_header skip (below), independent of freelist indexing.

**(b) `alignas(64)`, NOT `alignas(KAME_CACHE_LINE)`.**  The embed object
sits at `chunk_base + ALLOC_CHUNK_HEADER` (= +64) and chunk_base is
256 KiB-aligned, so the object is EXACTLY 64-aligned.  `alignas(128)`
(the Apple-Silicon KAME_CACHE_LINE) on a member makes the type
128-aligned, but placement-new at a 64-aligned address is UB and the
member would not truly be 128-aligned.  `alignas(64)` is the placement
ceiling — and it SUFFICES: because the object already starts at +64,
aligning the hot block to 64 lands m_owner_id at object-offset 64 =
absolute chunk_base+128, a cache line distinct from chunk_header[0..63]
for BOTH 64 B (line 2) and 128 B (line 1) targets.  Verified by probe:

    alignof(PoolAllocatorBase)=64  sizeof=512
    m_owner_id @64  m_fs_flag @68  m_base_bucket @70  m_freelist_head @72
    -> absolute +128 : 64B line idx 2, 128B line idx 1 (chunk_header=line0)

(256 B lines (Fujitsu) can't be cleared without 256-aligning the object,
i.e. moving ALLOC_CHUNK_HEADER — out of scope; harmless, only a missed
false-sharing win there.)

**Released signal.**  `deallocate_chunk` now also clears
`chunk_obj->m_owner_id = 0` (plain store, published by the same claim-bit
release as palloc/size_info).  The fast path's owner-id compare thus
rejects a released chunk WITHOUT reading palloc; palloc==0 stays the
slow-path released-check.  The chunk is empty at release (live-slot
invariant) so no concurrent fast-path dealloc can race the clear.

**Resulting fast path (deallocate, allocator.cpp):**

    chunk_base resolved via s_back_offset (no chunk_header read)
    chunk_obj = (PoolAllocatorBase*)(chunk_base + ALLOC_CHUNK_HEADER)
    if (chunk_obj->m_owner_id == s_tls_owner_id) {        // cache line 1
        bucket = chunk_obj->m_fs_flag ? chunk_obj->m_base_bucket   // FS=true
                                      : *(uint32_t*)(p-8);         // FS=false
        if (FS=false && bucket out of range) goto slow;   // defensive
        chunk_obj->freelist_push(bucket, p);              // line 1 (small b)
        return true;
    }
    slow: palloc = *(chunk_base+8); if (palloc<=1) return false; vtable fn

chunk_header's cache line 0 (palloc @+8, size_info @+0) is NEVER read on
the owner-free fast path.

**Verification (x86-64, Release/-O2):** probe offsets as above;
alloc_stress cross=100 x13 (slow/cross path) + cross=0/5 x12 (owner-free
fast path) all PASS (sentinel_fails=0, diff=0); alloc_bucket34_repro
(FS=false ALIGN=256) PASS; c_api_test PASS.  alloc_minimal_bench hot
alloc+free vs "(1)" baseline f3d90835: 64 B (FS=true) 159->164 M ops/s,
**256 B (FS=false) 148->163 M ops/s (~+10%)** — the FS=false gain is the
chunk_header size_info read removed from line 0.  (The 128 B false-share
win is by construction; not measurable on this 64 B-line host.)

### 12.3 LANDED — direct-jump TLS shortcut + compact local-id freelist + retired g_thread_chunks

Two follow-on refinements on top of "(1b)" / §12.2 (`6a355619`):

**(a) Compact LOCAL-id-indexed per-chunk freelist.**  `m_freelist_head[]`
in the chunk is now sized `[KAME_LOCAL_BUCKETS = 9]` (down from the
global `[48]`) and indexed by a per-chunk LOCAL id = bucket's position
in its `(ALIGN,FS)` template's size set:
  - FS=true:       1 size per chunk -> id 0
  - ALIGN=32  FS=false: {6,8,10,12,14}    -> 0..4
  - ALIGN=64  FS=false: {24..31}          -> 0..7
  - ALIGN=256 FS=false: {16, 32..39}      -> 0, 1..8
  - ALIGN=1024 FS=false: {40..47}         -> 0..7
`kBucketLocalId[48]` (allocator_prv.h) is the global->local map.  On a
128 B cache line (Apple Silicon) the whole hot block — m_owner_id,
m_fs_flag, all 9 freelist heads — fits in ONE line.  The FS=false slot
prefix STORES the local-id directly (low 32 b of the `{local_id, SIZE}`
8 B header at p-8) so dealloc reads it with no remap.

**(b) Direct-jump TLS shortcut: `g_thread_freelist_ptr[bucket]`.**  A
new per-thread TLS array of `char **` — each entry points DIRECTLY at
the active chunk's `m_freelist_head[kBucketLocalId[bucket]]` cell.
Maintained by `slow_allocate` and `bucket_first_access` (the COLD paths)
where `kBucketLocalId[]` IS read.  The alloc hot path becomes ONE TLS
read + one indirect deref:

    if(char **head_ptr = g_thread_freelist_ptr[bucket]) {
        if(char *head = *head_ptr) {
            *head_ptr = *(char**)head;
            return head;
        }
        return chunk_from_freelist_ptr(head_ptr)->slow_allocate(bucket, size);
    }
    return cold_first_access(bucket, size);

— NO bucket->local-id remap on the hot path, NO chunk-pointer-deref
chain.  Dealloc hot path is unchanged in spirit (uses local-id directly
from `m_fs_flag` for FS=true or the slot prefix for FS=false to index
`chunk_obj->m_freelist_head[local]`).

**(c) `g_thread_chunks[]` RETIRED.**  The chunk pointer is recoverable
from any `g_thread_freelist_ptr[bucket]` entry via a single mask:

    inline PoolAllocatorBase *chunk_from_freelist_ptr(char **fp) {
        uintptr_t cb = (uintptr_t)fp & ~(ALLOC_MIN_CHUNK_SIZE - 1);
        return (PoolAllocatorBase*)(cb + ALLOC_CHUNK_HEADER);
    }

(chunks are `ALLOC_MIN_CHUNK_SIZE`-aligned; the embed object lives at
`chunk_base + ALLOC_CHUNK_HEADER` in the first unit, where `fp`
resides).  Used by new_redirected's slow branch + the chunk-release
sweep + bucket_first_access.  Removed: `g_thread_chunks[ALLOC_NUM_BUCKETS]`
TLS, its macOS TSD-fast setup (`s_kame_chunks_tsd_offset`,
`s_kame_chunks_key`, `kame_chunks_cold`, `kame_chunks_base`).  Saves
384 B / thread + half the TSD sentinel-scan work at process start.

**`new_redirected_large` properly fixed.**  Its earlier
`kame_slots_base()[bucket]` pop was ORPHAN code from before (1) — nothing
pushed to `g_thread_slots[]` after the rework, so the freelist was
always-empty there.  It now uses the same `g_thread_freelist_ptr[]`
direct-jump as `new_redirected`, sharing the same storage owner-side
dealloc writes to.

**Verification (x86-64 Linux, -O2):** alloc_stress 2000 threads × 4 runs
× cross={0,5,100} all PASS (sentinel_fails=0, diff=0); alloc_bucket34_repro
(FS=false ALIGN=256, exercises multi-local-id-per-chunk + prefix-stores-
local-id) PASS; c_api_test PASS.  alloc_minimal_bench hot vs `6a355619`
(global-[48] (1b)) 16-iter interleaved A/B: 64 B FS=true delta -1.5%,
256 B FS=false delta -1.2% — within microbench noise.  The compact
[KAME_LOCAL_BUCKETS=9] hot block + retired g_thread_chunks reduces
cache-line touches on 128 B-line targets (Apple Silicon) by construction;
not measurable on this 64 B-line host.

### 12.4 LANDED — initial-exec TLS for the hot path (no more __tls_get_addr)

Perf profile of `b428a62e` (§12.3 direct-jump compact) showed
`__tls_get_addr` at ~15% of total runtime in BOTH `operator new[]` and
`PoolAllocatorBase::deallocate`.  Cause: kamepoolalloc.so is built as a
shared library, and `__thread` defaults to the **global-dynamic** TLS
model on shared libs, which lowers each access to a libc thunk call.
The hot-path assembly outside the thunk was already minimal (7-8
instructions); the thunk WAS the bottleneck.

Switched the small hot TLS variables to **`initial-exec`** via a new
`ALLOC_TLS_IE` macro that adds
`__attribute__((tls_model("initial-exec")))`:
  - `g_thread_freelist_ptr[48]` (384 B): alloc fast path
  - `s_tls_owner_id`           (4 B):   dealloc fast path
  - Total IE static-TLS demand: ~400 B (well under Linux's ~4 KiB
    surplus-static-TLS budget for shared libs).

Larger cold TLS (`tls_cross_dealloc_batch` 16 KiB, per-template `s_tls`)
stays on global-dynamic — they tolerate the thunk and would crowd out
the static-TLS budget under IE.

**Constraint:** IE-marked variables can no longer be `dlopen`'d after
process start.  kamepoolalloc must be loaded at startup via
`LD_PRELOAD` or normal `-l` link.

**Effect on the hot path (operator new[], FS=true, 64 B):**

    # Before (global-dynamic): 25+ insns including the call
    lea     <descriptor>(%rip),%rdi
    call    __tls_get_addr@plt        # ~15 cycles + GOT roundtrip
    mov     (%rax,%rdx,8),%rax        # base[bucket]
    ...

    # After (initial-exec): single compound-addressing TLS load
    mov     <offset>(%rip),%rax       # GOT-resolved IE descriptor (L1-hot)
    mov     %fs:(%rax,%rdx,8),%rax    # ★ TLS load in ONE instruction

Identical compactness for the `mov %fs:(%rax),%eax` load of
`s_tls_owner_id` on the dealloc fast path.  No stack frame needed in
either function.

**Bench A/B (alloc_minimal_bench hot, 16-iter interleaved averages):**

    size=64 B  (FS=true) : 161.79 -> 199.73 M ops/s   (+23.5 %)
    size=256 B (FS=false): 161.29 -> 199.21 M ops/s   (+23.5 %)

(Multi-thread alloc_stress also gains: cross=0 ~12 M -> ~17 M ops/s.)

**Verification:** c_api_test PASS; alloc_bucket34_repro
(sentinel_fails=0); alloc_stress cross={0,5,100} x 3 runs each, 2000
threads, all PASS (sentinel_fails=0, diff=0).  perf annotate confirms
zero `__tls_get_addr` calls in `operator new[]` or
`PoolAllocatorBase::deallocate`.

## 13. 2-level radix tree for O(1) pointer-to-region lookup

**Status: LANDED (step 1 of the HPC-scaling plan).**  Motivation: the
former `lookup_chunk(p)` / `deallocate(p)` / `size_of(p)` linearly
scanned `s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES]` to find a pointer's
owning region — O(populated regions) per call.  At the current 3200-cap
(100 GiB VA), worst-case scans are 3200 entries / dealloc; on the road
to HPC region counts (32K+ for ≥ 1 TiB) the linear walk would dominate.
Replaces the walk with a 2-level radix indexed by upper bits of `p`.

### Layout

  region index = `p >> ALLOC_MIN_MMAP_SHIFT`   (= 22 bits for 47-bit VA)
  L1 [11 bits, 2048 entries × 8 B = 16 KiB BSS]  atomic<RadixL2Node *>
  L2 [11 bits, 2048 entries × 4 B = 8 KiB / node]  atomic<uint32_t>
       (0 = unpopulated; non-zero = `ccnt + 1`)

L2 nodes are allocated LAZILY via direct `mmap` (NOT libc malloc — we
interpose it; recursing through `kame_malloc` → `allocate_chunk` →
`radix_insert` → `libc malloc` would loop).  Each L2 node covers
`2^11 × 32 MiB = 64 GiB` of VA, so a 1-TiB-populated pool uses ~16 L2
nodes (128 KiB committed).

### Critical correctness fix — region alignment

The old mmap claim path aligned regions to `ALLOC_MAX_CHUNK_SIZE = 1 MiB`,
NOT `ALLOC_MIN_MMAP_SIZE = 32 MiB`.  So a region's 32 MiB VA range could
SPAN two radix slots — upper-portion pointers would miss the lookup.
**Fix:** bump `kAlign` to `ALLOC_MIN_MMAP_SIZE`.  Cost: ≤ 32 MiB of VA
per region (negligible).

### Concurrency

  - L2 leaf install: 1 CAS on the L1 slot; loser munmaps its leaf.
  - L2 slot store: serialized (region claim winner is unique via the
    `s_mmapped_spaces[ccnt]` CAS) so no contention.
  - Readers: acquire on the L1 slot synchronizes-with release stores of
    the L2 slot (and the slot's init store).  Wait-free.  No
    reclamation — slots are persistent (regions don't unmap currently).

### Per-thread 1-entry cache

The radix's two chained dependent loads add ~5 cycles per dealloc vs the
old "linear walk over 1-2 regions" hot path.  Restore most of that with
a per-thread 1-entry cache (`s_last_region_base` / `s_last_region_ccnt`,
IE TLS).  Updated on slow-path misses; locality-rich workloads (most
allocators) hit it nearly every dealloc.  Hit cost: 2 IE TLS loads +
compare = ~3 cycles.  Miss falls to out-of-line `radix_lookup_slow`.

### Derived `mp` skips `s_mmapped_spaces[ccnt]`

With 32-MiB region alignment, `mp = p & ~(ALLOC_MIN_MMAP_SIZE-1)`.  No
need to load `s_mmapped_spaces[ccnt]` on the hot path — one less mem op
per dealloc.  The array stays populated for the cold paths
(allocate_chunk / deallocate_chunk).

### Resulting fast path (deallocate, after `if(!p) return false`)

    base = p & ~MMASK
    if (base == TLS s_last_region_base)         # cache check, 2 IE TLS reads
        ccnt = TLS s_last_region_ccnt
    else
        ccnt = radix_lookup_slow(p)             # full radix walk + cache update
    if (ccnt < 0) return false
    mp = p & ~MMASK                              # derived, no array load
    pdiff = p - mp
    ... (rest of dealloc body unchanged)

### Bench (alloc_minimal_bench hot, 12-iter interleaved averages, x86-64)

  size=64 B  (FS=true,  1 region): 269 → 252 M ops/s   (−6.3 %)
  size=256 B (FS=false, 2 region): 262 → 267 M ops/s   (+1.9 %)

The 64-B regression is the cost of adding the radix cache check to the
hot path of a microbench that only ever touches one region (the
worst-case for any indexing scheme — there's nothing to scan).  At HPC
scale (1000+ regions) the old linear walk would have cost 1000 × 3
cycles per dealloc; radix stays at ~3 cycles.

### Verification

`c_api_test` PASS; `alloc_bucket34_repro` PASS (`sentinel_fails=0`);
`alloc_stress` cross={0,5,100} × 2 × 1500 threads PASS
(`sentinel_fails=0`, `diff=0`).

### Future (step 2)

`s_claim_bitmap`, `s_back_offset`, `s_region_has_free` move into each
region's first 256 KiB unit (per-region metadata in the region itself).
That + a region DLL retires `ALLOC_MAX_MMAP_ENTRIES` entirely — only VA
limits the region count.  Radix from step 1 stays as the lookup index.

## 13.2 Per-region metadata in unit 0 (HPC scaling step 2)

Builds on §13's radix.  Moves the two globals that scaled with
`ALLOC_MAX_MMAP_ENTRIES` — `s_claim_bitmap[]` and `s_back_offset[]` —
into a `RegionMeta` block embedded at `region_base + 0` (inside unit 0
of each mmap region).

### Layout

```
class PoolAllocatorBase {
    struct RegionMeta {
        std::atomic<BitmapWord> claim_bitmap[BITMAP_WORDS_PER_REGION];  //  16 B
        std::uint8_t            back_offset[NUM_ALLOCATORS_IN_SPACE];   // 128 B
    };  // 144 B total
    static RegionMeta *region_meta(char *mp) noexcept {
        return reinterpret_cast<RegionMeta *>(mp);
    }
    // s_claim_bitmap[]  retired
    // s_back_offset[]   retired
};
```

Region base is `ALLOC_MIN_MMAP_SIZE`-aligned (= 32 MiB, §13's
alignment fix), so the cast in `region_meta(mp)` is always valid for
populated regions.  The 144 B metadata lives in the first 4 KiB page of
unit 0; the remaining ~252 KiB of unit 0 is reserved (virtual only — no
physical commit).

### Reserved unit 0 + bit-0 hardwired

Allocate_chunk's bitmap scan must NOT give out unit 0 (the metadata
lives there).  Solution: at region init, set bit 0 of
`claim_bitmap[0]` to `1`.  The CAS-with-stride scan in
`try_claim_in_region` naturally skips position 0 because the bit is
already set.

Init happens BEFORE `s_mmapped_spaces[region]` is published via its
release-CAS; the CAS's release pairs with any subsequent reader's
acquire chain (via the radix's L1 entry release / read paths) to make
the bit-0 store visible.

Capacity cost: 1 unit per region = 0.78 %.  Additional 1 unit lost to
stride alignment for multi-unit chunks (CHUNK_UNITS=2: 1 of 64 unit
slots in word 0 wasted; CHUNK_UNITS=4: 3 of 64 wasted).  All trivial.

### `count_live_chunks` adjustment

The pre-§13.2 diagnostic counted set bits in `s_claim_bitmap`, returning
"units occupied by chunks."  Post-§13.2 the metadata bit would inflate
the count by one per populated region (90 regions populated would
report ~90 even with zero chunks live).  Fix: mask off bit 0 of word 0
in the per-region sum, restoring the original leak-probe semantics.

### Changes

  - `s_claim_bitmap[]` removed; access goes through `region_meta(mp)`.
  - `s_back_offset[]` removed; same.
  - `resolve_chunk_from_slot`: reads `rmeta->back_offset[unit_idx]`
    instead of the global.  `meta_base` parameter is now unused (kept
    as no-op for call-site compat).
  - `try_claim_in_region` (allocate_chunk template): grabs
    `rmeta = region_meta(s_mmapped_spaces[region])` once per call.
  - `claim_chunk` (runtime variant for dedicated large chunks): same.
  - `deallocate_chunk`: clears bits via `rmeta`.
  - Dealloc fast path: drops the `meta_base = ccnt * NUM_ALLOCATORS`
    multiply — `rmeta = region_meta(mp)` is a no-op cast.
  - Region claim path (both `allocate_chunk` and `claim_chunk`):
    initializes `rmeta->claim_bitmap[0] = 1` before publishing the
    region.

### Benefits

  - **~450 KiB BSS retired**: 3200 × (16 + 128) on 64-bit.  Replaced
    by per-region commit of 1 page (4 KiB) per populated region —
    scales with usage, not with the cap.
  - **Pre-step for retiring `ALLOC_MAX_MMAP_ENTRIES`**: the two
    biggest cap-sized arrays are now per-region.  Still capped on
    `s_mmapped_spaces[]` and `s_region_has_free[]`; step 3 handles
    those.

### Verification

  - x86-64 perf (alloc_minimal_bench hot, 12-iter interleaved):
      64 B  (FS=true,  1 region): 274 → 270 M ops/s  (−1.5 %, noise)
      256 B (FS=false, 2 region): 282 → 284 M ops/s  (+0.7 %)
  - x86-64 correctness: c_api_test, alloc_bucket34_repro
    (sentinel_fails=0), alloc_stress cross={0,5,100} × 3 × 1500 threads
    (sentinel_fails=0, diff=0), 9/9 PASS.
  - 32-bit (gcc-multilib): clean build, c_api + stress PASS.

## 13.3 Retire ALLOC_MAX_MMAP_ENTRIES via a push-only region list (HPC step 3)

Final step: remove the last two cap-sized globals — `s_mmapped_spaces[]`
(8 B/region) and `s_region_has_free[]` (1 bit/region) — so the region
count is bounded only by VA (the radix covers 47-bit = 128 TiB), not by
a fixed array.

### Key enabler: regions are permanent

Regions are never unmapped in the current design, so the region list can
be **push-only** (a lock-free Treiber push, never remove).  Walkers never
see a freed node; no ABA, no reclamation.

### Changes

  - `RegionMeta` (embedded at region_base+0, §13.2) gains:
      `std::atomic<RegionMeta*> dll_next;`   // push-only list link
      `std::atomic<unsigned char> has_free;` // was a global bitmap bit
  - Globals: `s_region_dll_head` (list head) + `s_region_count` (live
    count, for the cap + O(1) populated_region_count).
  - `s_mmapped_spaces[]` RETIRED — region base derived from any pointer
    (`p & ~(ALLOC_MIN_MMAP_SIZE-1)`, regions are 32-MiB-aligned), via the
    new `region_meta_of(p)`.  Enumeration walks the list.
  - `s_region_has_free[]` RETIRED — per-region `has_free` flag + list walk.
  - radix slot is now a pure **presence token** (1 = present); the region
    id (`ccnt`) is gone — callers derive the base from the pointer.  The
    per-thread cache drops its `ccnt` field (a hit just means present).
  - `mmap_new_region()` — shared helper (was duplicated mmap blocks in
    both claim paths): cap-reserve (atomic), 32-MiB-aligned mmap, init
    RegionMeta (reserve unit 0, has_free=1), `radix_insert`, list push.
  - `allocate_chunk<ALLOC>` and `claim_chunk` Pass 1 = list walk, Pass 2
    = `mmap_new_region`.  `deallocate_chunk` derives the region directly
    (no scan); `count_live_chunks` / `populated_region_count` walk the
    list / read the counter.
  - `ALLOC_MAX_MMAP_ENTRIES` -> `ALLOC_MAX_REGIONS`: no longer an array
    bound, just the default (uncapped) `s_max_regions_cap`.  64-bit:
    `1 << RADIX_REGION_BITS` = 4194304 (= 128 TiB, the radix ceiling,
    static_assert'd).  32-bit: 96 (3 GiB user-VA), kept small so the
    `regions × 32 MiB` byte math can't overflow a 32-bit size_t.

### Correctness fix found in testing — fresh-region swarm

A region we mmap is published to the shared list IMMEDIATELY, so under
high thread count other threads can claim all 127 of its units before
our own `try_claim` runs.  The first cut returned 0 on that miss →
spurious `std::bad_alloc` (seen ~2/15 at cross=100, 1500 threads).  Fix:
**both claim paths now loop** (Pass 1 → Pass 2 → retry), exactly as the
old index `for`-loop implicitly retried.  Terminates because each Pass 2
advances `s_region_count` toward the cap.

### Benefits

  - **Cap lifted: 100 GiB → VA-limited (128 TiB)** on 64-bit.  Stress
    now sustains peaks > 3200 regions (the old hard cap) — verified
    `peak=3248` where the old build would have thrown bad_alloc.
  - Last cap-sized BSS gone; region state is fully self-contained.
  - `populated_region_count` / `kame_pool_reserved_bytes` now O(1).
  - `deallocate_chunk` no longer scans regions (O(N) → O(1) derive).

### Verification

  - x86-64: c_api_test, alloc_bucket34_repro (sentinel_fails=0,
    chunks=0→2), alloc_stress cross={0,5,100}: 12/12 + an extra 10/10 at
    cross=100 (the swarm case), all sentinel_fails=0 / diff=0.  Region
    count probe via the C API: 20000×64 B → 1 region (32 MiB), stays
    mapped after free (regions permanent), reserved_bytes correct.
  - 32-bit (gcc-multilib): clean build, c_api + stress cross={0,100} PASS.
  - perf (alloc_minimal_bench hot, 12-iter interleaved vs §13.2): no
    regression (64 B / 256 B both within noise, trending faster — smaller
    BSS / better cache).
