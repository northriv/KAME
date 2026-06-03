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

## §14 Follow-ups (post-HPC-scaling)

### §14B — Opt-in transparent hugepages (`KAME_POOL_HUGEPAGE=1`)
Linux `MADV_HUGEPAGE` on the slot range (skipping the metadata page) of
every fresh 32-MiB region.  Default OFF because the single-thread
microbenchmark regresses on FS=false sizes (the kernel zero-fills a 2
MiB hugepage for chunks that touch only ~500 KiB, no TLB-pressure
payback at that scale).  Real HPC workloads with large aggregate
working sets benefit.  Verified by `/proc/$pid/smaps` `VmFlags: hg`.

### §14C — NUMA-aware region claim
- `RegionMeta` gains `numa_node` (16 b) — the node a region is bound to.
- `s_region_dll_heads[KAME_MAX_NUMA_NODES = 16]` replaces the single
  push-only list; each region pushes onto its creator thread's local
  node's list.
- `mmap_new_region` calls `mbind(MPOL_BIND, {my_node})` to bind the new
  region to the creator's NUMA node (no libnuma dependency — direct
  syscall).  Skipped when `s_num_numa_nodes <= 1` (no-op on single-node
  systems and on macOS/Windows).
- Claim-side Pass 1 walks the LOCAL node's list first, falls back to
  other nodes ((my_node + 1) % N, etc.).  Locality preserved on
  multi-node systems; identical traversal cost on single-node.
- `numa_node_for_this_thread()` lazy-inits a per-thread preferred node
  from `sched_getcpu()` + `/sys/devices/system/cpu/cpuN/nodeM`.  Marked
  `cold, noinline` so the inlined Pass-1 walker stays compact in the
  caller's I-cache.

### §14E — `kame_pool_get_stats()` observability
Versioned C-ABI struct returning regions_populated / bytes_reserved /
chunks_live / units_live.  Walks the region list under relaxed loads
(diagnostic-grade, not hot-path).

## 15. Forward-shift: slot region starts at 256 KiB unit boundary

Shift every chunk's start by `K_MAX` (= 4 KiB, one page) bytes backwards
from its first claimed unit boundary, so the slot region begins exactly
at the unit boundary.  This gives every slot 0 a deterministic 256 KiB
alignment for downstream consumers (SIMD, DMA, THP, etc.).

### Layout

```
unit_boundary[base_unit] - K_MAX = chunk_base[N]
   [chunk_base + 0..63]      chunk_header (palloc, fn, sizeof_fn, size_info, dedicated size)
   [chunk_base + 64..]       PoolAllocator object
   [...]                     m_flags[count]
   [...padding...]
unit_boundary[base_unit] = chunk_base[N] + K_MAX
   [chunk_base + K_MAX ..]   slot region (256 KiB-aligned)
   [...]                     slots × ALIGN bytes
chunk_base + chunk_size - K_MAX
   [last K_MAX of chunk]     reserved for chunk N+1's metadata (if any)
chunk_base + chunk_size = unit_boundary[next_chunk_base_unit] - K_MAX
```

Adjacent chunks tile end-to-end — chunk N's last K_MAX bytes are
exactly chunk N+1's metadata (if a next chunk is later claimed).
First chunk in a region (base_unit = 1) has its metadata in unit 0's
last K_MAX bytes (RegionMeta lives at unit 0's start, ~150 B).

### Implementation

- `ALLOC_CHUNK_K_MAX = 4096` (one OS page on x86-64 Linux).
- `PoolAllocator::create()` (both FS=true and FS=false): slot region
  position fixed at `ppool + K_MAX - ALLOC_CHUNK_HEADER` (= chunk_base
  + K_MAX = unit_boundary).  Slot count selected to satisfy both
  metadata-fit and slot-region-fit constraints.
- `allocate_chunk` / `claim_chunk`: chunk_base = region_base +
  base_unit_idx × 256 KiB − K_MAX.
- `resolve_chunk_from_slot` / `deallocate(p)` fast path:
  `chunk_base = mp + base_idx × 256 KiB − K_MAX`.
- `deallocate_chunk(chunk_base, ...)`: `base_unit_idx = (chunk_base
  + K_MAX) & MMASK >> ALLOC_MIN_CHUNK_SHIFT`.
- `chunk_from_freelist_ptr(fp)`: needed `+ K_MAX` before masking, then
  `− K_MAX + ALLOC_CHUNK_HEADER` to recover PoolAllocator object.  fp
  now sits in the previous unit's last page (PoolAllocator at
  chunk_base + 64 = unit_boundary − K_MAX + 64), so the original
  "mask fp to 256 KiB" trick aliased to unit U-1's boundary — caught
  in stress testing (multi-thread alloc SEGV on first dealloc).

### Slot capacity

Every chunk's slot region is `chunk_size − K_MAX` bytes (last K_MAX
reserved).  Loss per chunk:
  - 256 KiB chunk: 4 KiB / 256 KiB = 1.5 %
  - 512 KiB chunk: 0.8 %
  - 1 MiB chunk:   0.4 %
  - 4 MiB dedicated: 0.1 %
Negligible.

### Verification

x86-64 (Linux, 4 KiB page):
  - All correctness tests pass (c_api, bucket34, alloc_stress
    cross={0,5,100} × 3, 9/9 sentinel_fails=0 / diff=0).
  - Probe confirms slot 0 of every chunk is exactly 256 KiB-aligned:
    `kame_pool_malloc(64)` → returned pointer `mod 256K = 0`.
  - Perf (12-iter interleaved, vs §14C 5762d409):
      16  B: 246 → 253 M ops/s (+2.9 %)
      64  B: 244 → 253 M ops/s (+3.7 %)
      128 B: 243 → 252 M ops/s (+3.8 %)
      256 B: 243 → 252 M ops/s (+3.7 %)
      1024 B: 193 → 197 M ops/s (+2.0 %)
      1500 B: 193 → 198 M ops/s (+2.3 %)
      4096 B: 192 → 193 M ops/s (+0.8 %)
      8192 B: 190 → 192 M ops/s (+1.0 %)
    Better than the predicted "≈ 0 %".  Likely from cache-line / DC
    miss predictability when chunk's slot region pages align with
    natural prefetcher / page-walker assumptions.

32-bit (gcc-multilib): clean build, c_api + stress 4/4 PASS.

### Caveats

- ALLOC_PAGE_SIZE > 4096 (Apple Silicon 16 KiB, PowerPC 64 KiB):
  K_MAX = 4096 is smaller than a page, so the madvise in
  `deallocate_chunk` may round up and reclaim a few extra KiB of slot
  region's first page along with the metadata.  Correctness preserved
  (palloc=0 is set BEFORE madvise; if madvise touches it the chunk is
  already released).  Memory waste is < 1 page per chunk-release on
  those platforms.

### Follow-up fix: dedicated large chunks under §15

`allocate_dedicated_chunk` (the 17 KB .. 4 MiB single-slot path from
§04d58d06) was NOT updated when §15 shifted every chunk back by K_MAX.
It still returned `chunk_base + ALLOC_CHUNK_HEADER` (the pre-§15 payload
start, header-at-front layout).  Under §15 `chunk_base = unit_boundary
- K_MAX`, so `chunk_base + 64` lands in the PREVIOUS unit; on free,
`deallocate(p)` computes `unit_idx = (p - mp) >> 18 = base_unit - 1`,
reads `back_offset[base_unit - 1]` (a neighbouring chunk's entry, or 0),
and resolves a bogus chunk_base → SIGSEGV in `PoolAllocatorBase::
deallocate`.  Reproduced by `alloc_minimal_bench 32768 100` (any size
> ALLOC_MAX_BUCKETED_SIZE that takes the dedicated path).

Fix — make the dedicated payload obey the same forward-shift layout as a
regular chunk's slot region:
  - payload starts at `chunk_base + K_MAX` (= unit boundary, 256 KiB-
    aligned), so `deallocate` resolves `unit_idx = base_unit`,
    `back_off = 0`, `base_idx = base_unit`, chunk_base correct.
  - units needed = `ceil((size + K_MAX) / 256K)` (was `+ HEADER`):
    payload reserves the trailing K_MAX of its last unit for the next
    chunk's metadata, exactly as a regular chunk reserves
    `chunk_size - K_MAX` for slots.
  - `size_of` (malloc_usable_size) for the dedicated sentinel returns
    `total - K_MAX` (was `- HEADER`).
  - `allocate_large_size_or_malloc`'s pool-vs-libc cutoff is
    `ALLOC_MAX_CHUNK_SIZE - K_MAX` (was `- HEADER`), matching the
    chunk_units cap.

Verification (x86-64, 4 KiB page):
  - `alloc_minimal_bench` at 32761, 50000, 100000, 1M, 4M (dedicated)
    and 4190209+ (libc fallback) — no SEGV.
  - usable_size exact: 258048 (1u), 1044480 (4u), 4190208 (16u, max) =
    `chunk_units*256K - K_MAX`.
  - byte-pattern integrity test (write full requested range, verify on
    free): PASS for all pooled dedicated sizes.
  - 16-thread × 4000-iter dedicated-chunk churn (sizes 17 K..3 MiB,
    per-allocation pattern verify): PASS 3/3.
  - c_api + alloc_stress: PASS.

## 16. Full-usable m_sizes mode for ALIGN >= 1024 (kill the 50 % page round-up)

The FS=false "borrow scheme" (§12.3) stores each slot's `{local_id, SIZE}`
prefix in the slot's own LAST 8 bytes, so a slot of N units has only
`N*ALIGN - 8` usable bytes.  For a request of exactly `M*ALIGN` (a
power-of-2 page multiple) this forces `N = M + 1` — and with the ALIGN=4096
tier's power-of-2 bucket set (N ∈ {1,2,4,8}) the +1 rounds up to the NEXT
power of two: a 4096-byte request lands in the 8192 slot, **50 % wasted**.
The 8-byte theft is also why a slot is never a clean page multiple, so the
tier could not serve page-aligned allocations efficiently.

Fix: for FS=false chunks with **ALIGN >= 1024** (buckets 40..51 — the
ALIGN=1024 and ALIGN=4096 tiers), move the per-slot metadata OUT of the
slot into a chunk-header `m_sizes[]` array, so the full `N*ALIGN` bytes are
user-usable and `N = ceil(SIZE/ALIGN)` (no `+8`).  Selected per template
via `if constexpr (ALIGN >= 1024)`; ALIGN < 1024 (small, hot) keeps the
borrow scheme and its (1b) cache discipline unchanged.

### m_sizes layout & encoding

- `uint16_t *m_sizes` lives in `PoolAllocatorBase`'s hot block (next to
  `m_owner_id` / `m_fs_flag`), null for borrow-mode chunks.  `uint8_t
  m_align_shift = log2(ALIGN)` sits beside it.
- The array is placed right after `m_flags[count]` in the chunk metadata
  region; `create()` reserves it in the count-selection budget
  (`per_word_meta = sizeof(FUINT) + FUINT_BITS*sizeof(uint16)` for the
  full tier).  For ALIGN=1024 (1 MiB chunk, count ≈ 15) that's 15×64×2 =
  1920 B, well under the K_MAX−64 = 4032 B budget; count stays
  slot-limited so the extra term never reduces capacity.
- Indexed by slot START bit: `bit = (p - m_mempool) >> m_align_shift`.
  Each entry packs `(N << 8) | local_id` — `local_id` (low byte) for the
  freelist index on the dealloc fast path, `N` (high byte) for the
  batch_return bit-clear and `size_of` (= `N*ALIGN`).

### Touch points (all `if constexpr (ALIGN >= 1024)` in the FS=false leaf)

- ctor: set `m_sizes`/`m_align_shift` (the `FS && !DUMMY` instantiation is
  the FS=false base subobject; real FS=true is `<ALIGN,true,true>`).
- `allocate_pooled`: `N = ceil(SIZE/ALIGN)`; write `m_sizes[bit]` instead
  of the `slot_start-8` prefix (before the publishing CAS — same
  release/acquire ordering carries it to a cross-thread freer).
- `deallocate_pooled` + `PoolAllocatorBase::deallocate` fast path: read
  `local` from `m_sizes[bit] & 0xFF` (the base path branches on the
  runtime `m_sizes != null`; borrow chunks see null in the already-loaded
  hot line and fall to `p-8` with no new cache line).
- `batch_return_to_bitmap` MaskFn: `N = m_sizes[bit] >> 8`.
- `size_of_static`: return `(m_sizes[bit] >> 8) * ALIGN`.
- `slow_allocate`: derive `slot_size` from `kBucketNewSlot[bucket]` (the
  4-way octave/sub formula does NOT cover the out-of-order page tier
  48..51 — this also fixes a latent SEGV that never fired because the
  alloc_minimal_bench freelist never misses).  Full tier passes
  `slot_size = slot`; borrow tier `slot - 8`.

### Bucket schedule & routing

Buckets 40..51 are now full-usable, `SIZE = N*ALIGN` (was `N*ALIGN - 8`):
ALIGN=1024 → 6144..17408, ALIGN=4096 → 4096/8192/16384/32768.
`ALLOC_MAX_BUCKETED_SIZE` 32760 → 32768.

`bucket_for_size` splits into two passes (`kame_ladder_bucket` helper):
borrow tier routes with `total = size + 8`, full tier (ALIGN>=1024)
recomputes with `total = size`.  Page-aligned overrides: 3833..4096 →
48 (vs borrow 39); 15361..16384 → 50 (vs full 47); 17409..32768 → 51.
Bucket 49 (8192) ties full bucket 42 (8192) — plain malloc stays on 42
(denser ALIGN=1024 chunks); 49 is reserved for a future large-alignment
`posix_memalign` / `aligned_alloc` path (currently those fall back to libc
for align > 16).

### Verification (x86-64 4 KiB page + 32-bit)

  - Sweep of all 39 632 sizes 369..40000: usable >= request, and writing
    the full requested range survives a neighbour alloc/free (no metadata
    overlap).  64-bit AND 32-bit (gcc -m32): PASS.
  - Page sizes 4096 / 8192 / 16384 / 32768 → usable EXACTLY 4096 / 8192 /
    16384 / 32768 (full slot — the 50 % round-up is gone; was 8192 usable
    for a 4096 request under borrow).
  - Page-aligned buckets 48/50/51 return 4096-aligned pointers (64/64).
  - 16-thread × 30 000-iter churn over 1 K..32 K holding 24 live/thread
    (forces freelist-miss `slow_allocate` + bitmap CAS + cross-thread
    frees), per-allocation pattern verify: PASS 4/4.
  - realloc grow/shrink across 500..100000 (full-usable + dedicated
    boundaries): data preserved.
  - Small-alloc hot path (16/64/128/256 B, borrow mode) unchanged:
    254..280 M ops/s (the base fast path's extra `m_sizes==null` branch is
    free — same cache line as m_owner_id).  Full-usable range hot path
    145..165 M ops/s.
  - c_api + alloc_stress + bucket34_repro: PASS (64-bit and 32-bit).

## 17. Pool-routed over-aligned allocation (POSIX)

Every pool slot is `kBucketAlign[bucket]`-aligned because the slot region
starts at a 256 KiB unit boundary — a multiple of every `kBucketAlign`
entry — and slot j is at `mempool + j*ALIGN`.  So we can serve
`posix_memalign(A, S)` / `aligned_alloc(A, S)` / `new(align_val_t{A})`
from the existing buckets whenever some bucket's ALIGN is a multiple of A
and its usable size covers S.  No `_aligned_free` pairing required — the
returned pointer is an ordinary pool slot freeable via the standard
`kame_free` path, with libc fallback handled transparently in
`PoolAllocatorBase::deallocate`.

### Routing

`bucket_for_aligned(A, S)` (cold path, linear scan of 52 buckets) picks
the smallest bucket with `kBucketAlign[b] >= A`, `kBucketAlign[b] % A == 0`,
and `kame_bucket_usable(b) >= S`:
  - A ∈ {32, 64, 256, 1024, 4096} → the matching ALIGN tier (32/64/256
    mid buckets and the 1024/4096 full-usable tiers).
  - No match (A > 4096, OR A ≤ 4096 but S exceeds every matching bucket):
    `new_redirected_aligned` falls back to `allocate_dedicated_chunk`
    (its §15 payload starts at a 256 KiB unit boundary — A-aligned for
    every A up to 256 KiB), and finally to libc `posix_memalign`.

### Entry points (POSIX)

`new_redirected_aligned(A, S)` — pool-or-libc dispatch; mirrors
`new_redirected_large`'s freelist / `slow_allocate` / `cold_first_access`
cascade for the chosen bucket.

  - C API: `kame_pool_aligned_alloc` and `kame_pool_posix_memalign` route
    A > 16 through it (≤ 16 stays on `new_redirected`).
  - C++: `operator new(size, align_val_t{A})` and its array / nothrow
    siblings.  Matching `operator delete(p, align_val_t{A})` calls
    `deallocate_pooled_or_free(p)`, which resolves pool pointers via
    `PoolAllocatorBase::deallocate` and libc-fallback pointers via libc
    `free` (same unified free as the rest of the pool).

Windows keeps the platform-native `_aligned_malloc` / `_aligned_free`
pairing because the pool free path has no alignment info to dispatch
`_aligned_free` correctly.

### Bug found during integration: `local_id` ambiguity at SIZE=8192

`allocate_pooled` derived the slot's `local_id` via
`kBucketLocalId[bucket_for_size(SIZE)]`.  For the full-usable tier this
breaks at SIZE = 8192: bucket 49 (ALIGN=4096 N=2) and bucket 42
(ALIGN=1024 N=8) share that slot size, and `bucket_for_size` routes a
plain 8192-byte request to bucket 42 (denser ALIGN=1024 chunks).
Reached via the §17 aligned path, an ALIGN=4096 chunk was tagging its
slot with bucket-42's local_id (2 instead of 1), and the eventual free
pushed the slot to the chunk's `m_freelist_head[2]` (bucket 50's list);
the next bucket-50 alloc popped the WRONG-size slot and overlapped a
neighbour — caught by the per-byte pattern verify at offset 8416 of a
10000-byte slot.  Fix: derive `local_id` directly from (ALIGN, N) inside
the full-usable branch of `allocate_pooled` (the borrow tier keeps the
table lookup, which IS bijective because user_max = N*ALIGN-8 uniquely
identifies the bucket).

### Verification

  - Aligned sweep — A ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096} × 45
    sizes spanning every bucket transition + 256 KiB-aligned bigger
    cases (4 A × 4 S): every returned pointer is A-aligned, usable ≥ S,
    full-range writes survive neighbour alloc/free.  376 combos PASS on
    64-bit and 32-bit.
  - Aligned MT stress: 16 threads × 20000 iter × random (A, S), 16 live
    per thread (forces freelist-miss / slow_allocate / cross-thread
    frees in the aligned path), per-allocation pattern verify: PASS 4/4.
  - C++ `new (align_val_t{N}) Foo` for N ∈ {64, 256, 4096}: pointers are
    N-aligned, delete frees through the unified pool path.  PASS.
  - Regression: §15 dedicated, §16 full-usable, c_api, alloc_stress,
    bucket34_repro: all PASS (64-bit and 32-bit).

## 18. OOM handling — noexcept-safe nullptr returns + `new_handler` retry

Pre-fix `PoolAllocator::create_allocator()` did `throw std::bad_alloc()`
on mmap region cap exhaustion / kernel mmap refusal.  The throw
propagated through `allocate_chunk_path` / `slow_allocate` / the
`new_redirected*` family and reached:

- `kame_pool_malloc` / `_calloc` / `_realloc` / `_aligned_alloc` /
  `_posix_memalign` — all `noexcept` C wrappers ⇒ `std::terminate()`.
- `operator new(size, std::nothrow_t)` / over-aligned nothrow variants —
  `noexcept` per spec ⇒ `std::terminate()`.

Plain `operator new(size)` could *return* nullptr (when `new_redirected`
landed on a non-throwing OOM path like `posix_memalign` failure), but
the implementation didn't check the return; the caller dereferenced
null — violation of [new.delete.single]'s "throw bad_alloc if request
cannot be satisfied".

Standards-conformance fix:

  1. **`create_allocator` returns nullptr on OOM**, not throws.  Single
     `fprintf(stderr, ...)` diagnostic kept so the OOM is observable in
     test logs.  `allocate_chunk_path` propagates the null upward.

  2. **`new_handler` retry loop helper** (anon-namespace
     `try_alloc_with_new_handler`):
     ```cpp
     for(;;) {
        if(void *p = alloc_fn()) return p;
        std::new_handler h = std::get_new_handler();
        if(!h) return nullptr;
        h();  // frees memory and returns OR throws bad_alloc
     }
     ```
     Per [new.delete.single] the standard `operator new` must do this.
     `kame_alloc_with_handler(size)` and
     `kame_aligned_alloc_with_handler(A,S)` are the size / aligned
     wrappers used by the throwing `operator new` family.

  3. **Throwing `operator new` variants** (`operator new(size)`,
     `operator new(size, align_val_t)`, array forms):
     - Call the `_with_handler` helper.
     - On final nullptr, `throw std::bad_alloc()` themselves.

  4. **Nothrow `operator new` variants** (`operator new(size,
     nothrow_t)`, aligned nothrow):
     - Wrap the throwing path in `try { ... } catch(...)` so a
       `new_handler` that throws still produces the noexcept-contracted
       nullptr return — the standard says nothrow new returns null if
       the implementation cannot satisfy the request.

  5. **C wrappers** (`kame_pool_malloc` etc.) keep strict libc malloc
     semantics: no `new_handler` (that's a C++ concept), just call
     `new_redirected` directly and set `errno = ENOMEM` on null.  Now
     that step (1) eliminates the bad_alloc throw, no try-block is
     needed at the noexcept boundary.

### Verification

  - **Pool cap exhaustion**: `kame_pool_set_max_bytes(32 MiB)` then
    1024-byte alloc loop — OOMs at i=24003 with `errno=ENOMEM`,
    `kame_pool_reserved_bytes() == 32 MiB`; reset cap, new alloc
    succeeds.  Diagnostic line "kamepoolalloc: OOM — chunk-claim
    failed for ALIGN=..." printed on stderr.
  - **`operator new` throws**: pool exhausted → `try { ::operator new(1024); }
    catch(std::bad_alloc &)` catches.
  - **`operator new(nothrow)`**: pool exhausted → returns nullptr, no
    throw.
  - **`new_handler` loop**: install handler that disarms itself after
    3 calls — handler is called 3 times, then `bad_alloc` thrown.
    Confirms the retry loop honours `set_new_handler` per spec.
  - **Sanitizers**: release + TSAN + UBSAN (-no-vptr) + ASan + 32-bit
    all PASS on c_api, alloc_stress, sweep, aligned_sweep, §15/§16/§17
    MT.  No regression.

## 19. Large-alloc tier — radix-registered mmap with real `munmap` on free

Closes the long-standing "pool never returns VA" gap.  Pool regions
(§13.3) are push-only by design; large allocations that don't fit the
4 MiB dedicated-chunk cap previously fell through to libc malloc.  §19
adds a middle tier for sizes 4 MiB – 32 MiB that owns its own
32-MiB-aligned mmap per alloc and **really `munmap`s** on free, so
long-running processes get their VA back.

### Routing

Three-tier above-bucket dispatch in `allocate_large_size_or_malloc`:
  1. `size ≤ ALLOC_MAX_CHUNK_SIZE − K_MAX` (≈ 4 MiB) → `allocate_dedicated_chunk`
     (§15): a multi-unit single-slot chunk inside the regular 32 MiB
     region pool.  Many such allocs share radix slots / NUMA hints / DLL
     state with regular bucket chunks — best locality for moderate-large
     sizes.
  2. `mmap_size ≤ ALLOC_MIN_MMAP_SIZE` (≈ 32 MiB) → `allocate_large_va`:
     one `mmap` of `round_up(size + PAGE, PAGE)` at a 32-MiB-aligned
     base, registered as a single `KAME_RADIX_LARGE` radix slot, served
     warm from the §25/§26 recycle cache.  Pays one radix slot per alloc
     (acceptable in the multi-MiB range) and **returns VA on free** via
     `munmap` when not recycled.
  3. (§27) `mmap_size > 32 MiB` → STILL `allocate_large_va`, but the
     `mmap` spans multiple 32-MiB radix regions and only the HEAD slot is
     registered.  Safe: the alloc's sole valid user pointer (`base + PAGE`)
     always resolves to the head slot; the tail slots are never standalone
     `radix_lookup` targets (interior-pointer lookup into one alloc is UB
     caller-side) and the OS keeps the whole span mapped so no other alloc
     can claim a tail slot's VA.  The huge tier BYPASSES the recycle cache
     (its `lrc_idx` log space tops out at 32 MiB; above that all sizes
     collapse to one slot whose only pop gate is `cached ≥ need`, with no
     upper bound → over-satisfaction / RSS pinning).  libc `malloc` is the
     fallback only when the `mmap` itself fails.

### Radix protocol extension

The radix slot value gains a third state.  Was 0/1 (absent/present), now
0/1/2 (`KAME_RADIX_ABSENT` / `KAME_RADIX_POOL` / `KAME_RADIX_LARGE`).
`radix_lookup` returns the kind directly (single load already; no extra
work).  `radix_insert` takes a `kind` arg; `radix_clear` CAS-zeros the
slot prior to munmap.  `radix_lookup_slow` updates the per-thread
`s_last_region_base` cache **only for pool kind** so a §19 base never
lingers in another thread's TLS after its munmap.

### `LargeAllocMeta`

Lives in the first page of the mmap region; user pointer is
`base + ALLOC_PAGE_SIZE`.  Fields: magic sentinel (debug only),
`alloc_size` (user-requested), `mmap_size` (for munmap), `numa_node`
(reserved for future bind).

`malloc_usable_size(p)` on a §19 pointer returns `mmap_size − PAGE`
(the actual usable slack, not the originally-requested `alloc_size`),
matching libc's `malloc_usable_size` convention so realloc-elision in
client code can grow in place across the page-rounded tail.

### Concurrency

Lock-free.  Insert is a release store; clear is a release store; the
existing radix lazy L2 allocation (CAS-install via `radix_alloc_l2`)
covers any new L1 entries.  A racing reader either sees the live slot
(reads valid meta — alloc/free both keep meta intact until after the
clear) or KAME_RADIX_ABSENT (falls through to libc free, matching libc's
behaviour for foreign pointers).

### Verification

  - **Range sweep** across all four tiers (bucket / dedicated / §19 /
    libc) — every alloc returns 4 K-aligned (or 32 M-aligned for §19),
    `malloc_usable_size ≥ requested`, full-range writes survive
    free/realloc cycles.
  - **`munmap` confirmed**: 1000 × 8 MiB alloc+free cycles without
    cumulative VA growth.  `kame_pool_reserved_bytes()` shows only the
    incidental small-alloc pool growth (32 MiB), not 8 GiB.
  - **§19 MT** (8 threads × 500 iter × random 5–30 MiB, pattern verify
    on first + last page): TSAN race-free, UBSAN clean, ASan clean.
  - **Regression**: c_api, alloc_stress (2000 thread × 20K ops),
    bucket34_repro, sweep_test (39632 sizes), aligned_sweep (376
    combos), §15/§16/§17 MT — all PASS on release, TSAN, UBSAN
    (−vptr), ASan, 32-bit.

### Known cost: bench-pattern penalty vs libc's arena cache

`alloc_minimal_bench` at size 5 MiB shows ~0.2 M ops/s for §19 vs
~42 M ops/s for libc's mmap-arena fallback (master pre-§19).  The
penalty is per-allocation `mmap + munmap` syscalls; libc holds an
arena cache that reuses freed mmap regions in tight-loop micro-benches.

Real workloads typically alloc, use for some time, then free — the
mmap overhead amortises away.  A follow-up could add a per-process /
per-NUMA-node LIFO cache of recently-freed §19 regions (bounded depth,
size-keyed) to recover the tight-loop pattern.  Out of scope here;
correctness is the §19 deliverable.

## 20. Cross-thread free vptr-after-release UB — caches dll-cursor fields pre-call

UBSAN (-fsanitize=undefined, vptr check enabled by default) flagged
three sites where a cross-thread free dereferenced `this` (or the
sibling `c`) AFTER `batch_return_to_bitmap` may have released the
chunk.  When the cross-free is the chunk's LAST live slot AND BIT_OWNED
is already clear (owner exited), `batch_return_to_bitmap` runs the
placement-new destructor on `*this` inside its `i_am_releaser` branch
and immediately calls `deallocate_chunk(cbase, csz)` — the bytes are
still mapped but the C++ object's lifetime has ended, so any access via
the typed pointer is UB.

The follow-on code reads `this->m_owner_dll_head_addr` (and an atomic
pointer field `this->m_owner_dll_force_walk_ptr`) to decide between
same-thread cursor reset and cross-thread force-walk hint.  These two
fields are write-once at chunk construction so reading them BEFORE the
batch call is sound and the cached values stay valid across destruction.

Fix at three sites:
  - `CrossDeallocBatch::push_direct(c, s)` (FS=true cross-thread).
  - `PoolAllocator<ALIGN,false>::deallocate_pooled(p)` (FS=false default
    path — this was the site UBSAN tripped on first in
    alloc_stress_test).
  - `PoolAllocator<ALIGN,true>::deallocate_pooled(p)` post-teardown
    branch (s_alloc_tls_off).

Each site now loads `m_owner_dll_head_addr` and (where used) the atomic
acquire-load of `m_owner_dll_force_walk_ptr` into local cached variables
BEFORE the `batch_return_to_bitmap` call.  The cached values are then
used for the cursor-reset / force-walk dispatch, never re-touching
`this`.  The atomic load happens-before semantics with respect to
owner-exit's release-store of nullptr are preserved.

### Verification

  - **UBSAN with vptr check** (NO `-fno-sanitize=vptr`): c_api,
    alloc_stress (2000 threads × 20K ops × 10 % cross), bucket34_repro,
    sweep_test (39632 sizes), aligned_sweep (376 combos), §15 dedicated
    MT, §16 full-usable MT, §17 aligned MT, §19 §19 MT — all PASS.
    UBSAN runs clean with default flags; `-fno-sanitize=vptr` no longer
    required.
  - Regression: release + TSAN + ASan + 32-bit also PASS on all of
    the above.

## 21. Thread-exit page reclaim (default ON) + §19 large-alloc recycle cache

Two RSS / perf refinements, no protocol change.

### 21a. Thread-exit madvise default ON

`release_dll_chunks_for_thread` released empty chunks with
`reclaim_pages=false` — the ONE place the release protocol skipped the
slot-region `madvise(MADV_DONTNEED)` (a perf optimisation: the madvise
was ~30 % of bench-style thread-teardown).  Mid-run releases always
madvise, so this was the only "RSS held until process exit" gap.

Now gated by `s_thread_exit_reclaim` (atomic int, default 1).  Default
behaviour madvise's on thread exit too, so a thread that allocates and
frees a working set, then exits, returns its RSS promptly.  Measured
(8 threads × 20000 × 200 B alloc-then-free-all-then-exit, 30 batches):
steady-state VmRSS 26 MiB → 5 MiB with reclaim on.  The prior fast
teardown is one call away: `kame_pool_set_thread_exit_reclaim(0)`.

### 21b. §19 large-alloc per-thread recycle cache

§19's per-allocation `mmap + munmap` made a tight large-alloc/free loop
~500× slower than libc's mmap-arena cache (microbench: 5 MiB at
~0.2 M ops/s vs libc ~42 M).  `LargeVaCache` is a per-thread LIFO of a
few recently-freed §19 regions (still mapped) keyed by mmap_size:

  - `allocate_large_va`: `pop_fit(mmap_size)` reuses a cached region
    whose size is in `[need, 2*need]` (re-register in radix, refresh
    meta) — zero syscalls on a hit.  Miss → `large_va_raw_map`.
  - `deallocate_large_va`: `radix_clear` (so a double-free routes to
    libc, never a torn cache), then `push` — keep mapped if it fits the
    cap, else `large_va_raw_unmap` now.
  - Bounded: `MAX_ENTRIES = 4`, `MAX_BYTES = 64 MiB` per thread.  Most
    threads never touch the large tier → empty cache → zero cost.  RSS
    held while cached is ≤ 64 MiB per large-allocating thread.
  - Drained at thread exit by the `thread_local LargeVaCache` dtor
    (munmap only — no pool ops, so TLS-destruction order is irrelevant).

Cached regions are radix-CLEARED while in the cache and re-registered on
reuse (Option B): a double-free of a cached pointer hits
`KAME_RADIX_ABSENT` → libc free → clean error, not cache corruption.

### Verification

  - Perf: tight large-alloc/free loop recovered to ~35–46 M ops/s
    (≈ libc arena), from §19's ~0.2 M ops/s.
  - Thread-exit drain: 3200 thread×alloc cycles (16 threads × 200
    batches × 5 × 8 MiB), VmRSS drift 0 MiB — no per-thread cache leak.
  - §19 MT (8 threads × 500 iter × random 5–30 MiB, pattern verify):
    TSAN race-free, ASan clean, UBSAN-full clean (release + each sanitizer).
  - Regression: release + TSAN + ASan + UBSAN-full + 32-bit all PASS on
    c_api, alloc_stress, sweep, aligned_sweep, §15/§16/§17/§19 MT.
