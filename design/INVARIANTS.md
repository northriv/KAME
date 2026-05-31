# kamepoolalloc — Structural Invariants

A catalogue of the invariants the lock-free pool allocator relies on for
correctness.  Purpose: when changing `allocator.cpp` / `allocator_prv.h`,
`grep` the field or `§`-tag you are about to touch, find the invariants that
constrain it, see **what breaks** if violated and **whether a machine check
guards it**.  This is the blast-radius map.

Each entry: **INV-N — statement** · *enforced where* · *breaks if violated* ·
*verification*.

Verification legend:
- **TLA+** — a TLC-checked model in `tests/tlaplus/` / `tests/CHUNK_CLAIM_TLA_NOTES.md`.
- **GenMC** — an RC11 model-check harness (`tests/cds/`).
- **test** — a runtime test in `tests/` exercises it (not exhaustive).
- **comment-only** — relied on, documented in code, but not machine-checked
  (← candidate for Stage 2 verification work).

---

## 1. Chunk geometry (§15)

Constants: `ALLOC_MIN_CHUNK_SIZE = 256 KiB` (one "unit"); `ALLOC_CHUNK_HEADER
= 64`; `ALLOC_CHUNK_K_MAX = 4096`; regions are `ALLOC_MIN_MMAP_SIZE = 32 MiB`
(= 128 units); `ALLOC_MAX_CHUNK_SIZE = 4 MiB` (16 units).

- **INV-1** — A chunk's `chunk_base = unit_boundary − ALLOC_CHUNK_K_MAX`; its
  payload/slot region starts at `chunk_base + K_MAX = unit_boundary` (256 KiB-
  aligned). · *allocate_dedicated_chunk, claim_chunk, the embed layout* ·
  *breaks: `deallocate` resolves the wrong unit → frees a neighbour chunk
  (the historical "size > 17400 SEGV")* · **test** (alloc_stress, bucket34).

- **INV-2** — The embedded `PoolAllocator` object lives at `chunk_base +
  ALLOC_CHUNK_HEADER (= +64)`; its vtable pointer is at offset 0 of that.
  `chunk_header[0..63]` holds size_info(0)/palloc(8)/fn(16)/sizeof_fn(24)/
  dedicated_size(32). · *the header offset macros* · *breaks: a slot write
  landing in [chunk_base, +64) corrupts the vtable → virtual-dispatch jump to
  garbage (the bucket34 class of crash)* · **test** (bucket34_repro).

- **INV-3** — `m_flags` (the per-chunk bitmap) is placed at `chunk_base + 64 +
  roundup(sizeof(PoolAllocator))`, and the full-usable `m_sizes[]` array (§16)
  after it; both must fit within `[+64, +K_MAX)` (the `kMetaBudget = K_MAX −
  HEADER = 4032` byte window). · *create() static_assert(size_alloc ≤
  kMetaBudget), count_meta cap* · *breaks: metadata spills into the slot
  region → bitmap/slot corruption* · **comment-only** (compile-time
  static_assert covers the object-fit; the dual count cap covers the rest).

- **INV-4 (§28, page-bounding)** — A chunk hands out **no slot in its final OS
  page**.  `create()` rounds `slot_region_size` down to a multiple of
  `ALLOC_PAGE_SIZE`, so the last page (which on `PAGE > K_MAX` targets — macOS
  16 KiB — is shared with the *next* chunk's header) never holds a live slot.
  · *create() FS=true and FS=false: `slot_region_size &= ~(ALLOC_PAGE_SIZE−1)`*
  · *breaks: a chunk-tail slot in a page that `deallocate_chunk`'s madvise
  can't reclaim without zeroing the neighbour header → the macOS straddle
  crash* · **test** (alloc_madvise_straddle_repro; no-op on Linux 4 KiB).

- **INV-5 (§28, madvise range)** — `deallocate_chunk`'s page reclaim covers
  only `[roundup(slot0,PAGE), rounddown(chunk_base+chunk_size,PAGE))` — never
  any chunk's header page. · *deallocate_chunk* · *breaks: macOS straddle —
  outward-rounded madvise zeroes an adjacent live chunk's vtable + m_flags
  (`MACOS_MADVISE_STRADDLE_CRASH.md`)* · **test** (alloc_madvise_straddle_repro).

## 2. Chunk-claim state machine — `m_flags_packed` (§13)

Layout: bit 31 = `BIT_OWNED`; bits 0..30 = `MASK_CNT` (live-slot count).  Bit
30 is intentionally unused (was a retired `BIT_RELEASED`).  `BIT_OWNER_EXITED`
is *modelled by* the cleared-`BIT_OWNED` state, not a separate bit.

- **INV-6** — Exactly one releaser per chunk.  Cross-thread last-slot free
  (`atomicDecAndTest` bringing the word to 0) and owner-exit
  (`atomicFetchAnd(~BIT_OWNED)` yielding 0) are mutually exclusive: while
  `BIT_OWNED` is set the dec-to-zero never fires (word stays ≥ `BIT_OWNED`);
  once cleared, the owner's AND-result==0 identifies it as releaser. ·
  *batch_clear_impl, release_dll_chunks_for_thread, owner_release* · *breaks:
  double-free or leak of a chunk* · **TLA+** (ChunkRecycle_microscopic,
  ChunkRecycle_threadepoch).

- **INV-7** — A chunk with `MASK_CNT > 0` is never released. · *owner_release
  refuses when MASK_CNT≠0; the exit walk only releases on newv==0* · *breaks:
  freeing a chunk with live slots → use-after-free* · **TLA+** (ChunkRecycle).

- **INV-8** — `BIT_OWNED` is set at chunk construction and cleared exactly
  once (by the owning thread's exit walk or owner_release). Cross-thread frees
  load it with `acquire`. · *PoolAllocator ctor, release paths* · *breaks:
  cross-thread releaser proceeds while owner still alive → race* · **TLA+**.

## 3. Region & radix (§13.2 / §13.3 / §19)

2-level radix `g_lrc... ` → `s_radix_l1[L1] → RadixL2Node.entries[L2]`, slot
value ∈ `{KAME_RADIX_ABSENT=0, KAME_RADIX_POOL=1, KAME_RADIX_LARGE=2}`.

- **INV-9** — A radix slot is written once at region/large-alloc publish
  (release store) and read with acquire; `radix_clear` CAS-zeros before any
  munmap. · *radix_insert/radix_lookup/radix_clear* · *breaks: a reader sees
  a stale kind and dispatches the wrong free path* · **comment-only**
  (← Stage 2: radix publish/clear ordering GenMC harness).

- **INV-10 (§27)** — A huge alloc (> LRC_HI) spans multiple 32-MiB radix
  regions but registers **only the head slot**.  Safe because the only valid
  user pointer is `base + PAGE` (always in the head slot); tail slots stay
  ABSENT and the OS keeps the whole span mapped so no other alloc can claim a
  tail slot's VA. · *allocate_large_va / deallocate_large_va* · *breaks:
  nothing observed — interior-pointer lookups are UB caller-side* · **test**
  (alloc_huge_test).

- **INV-11** — `RadixL2Node` member is named `entries`, **not** `slots` — Qt
  `#define slots` (qobjectdefs.h) would otherwise blank the array declaration.
  · *allocator_prv.h* · *breaks: compile error in Qt-including TUs* ·
  **test** (compiles in kame.app).

## 4. back_offset & dedicated-chunk dispatch (§15 / §22)

`s_back_offset[unit]` (one byte/unit/region): `base_u = u − back_off`,
`chunk_base = region + base_u·256K`.  Bit 7 (`0x80`) flags a dedicated chunk.

- **INV-12** — For any claimed slot, `back_offset[unit]` correctly recovers
  `chunk_base`; `deallocate_chunk` clears `back_offset[base..base+units)`
  *before* the claim-bit release. · *claim_chunk, deallocate_chunk* ·
  *breaks: free resolves the wrong chunk* · **TLA+** (ChunkRecycle_microscopic
  models the address arithmetic).

- **INV-13** — A dedicated chunk (256 KiB–4 MiB single alloc) is identified on
  the free fast path by `back_off & 0x80` alone (no chunk_header read). The
  total byte size lives at `chunk_header[32..39]`. · *PoolAllocatorBase::
  deallocate* · *breaks: dedicated free mis-routed as a bucket free* · **test**.

## 5. owner-id identity

- **INV-14** — `kame_owner_id()` is always non-zero (`s_owner_id_next` starts
  at 1, skips 0 on wrap). A released chunk has `m_owner_id == 0`; a foreign /
  never-allocated thread has `s_tls_owner_id == 0`. So the single comparison
  `chunk_obj->m_owner_id == s_tls_owner_id` subsumes the live + same-thread +
  not-foreign check on the dealloc fast path. · *kame_owner_id, deallocate
  fast path* · *breaks: a foreign or released chunk mis-identified as owned →
  wild write* · **test** (alloc_stress cross-thread free).

## 6. Large-recycle cache — L1/L2 (§21–§28)

K-major layout: `g_lrc[k].slots[idx]`, `k ∈ [0,LRC_K_MAX)`, `idx ∈
[0,LRC_N_MAX]`; each `LrcKArray` is `alignas(KAME_CACHE_LINE)`. Kinds:
`LRC_CHUNK` (idx [0,LRC_CHUNK_BND]), `LRC_MMAP` (idx (LRC_CHUNK_BND,N_MAX]).

- **INV-15** — Exclusive ownership: a block is taken from a slot by at most
  one thread.  `compare_exchange_weak(b→nullptr)` is the linearization point;
  losers move to the next k (no retry loop). · *global_pop_fit* · *breaks:
  two threads own the same block → double-free / UAF* · **GenMC**
  (cds_lrc_ownership.c).

- **INV-16** — No premature release: a block is `lrc_release`d
  (munmap/recycle_release_chunk) only by its current exclusive owner; size is
  read **after** the take-CAS (own-then-read), never peeked before ownership.
  · *global_pop_fit too-small path, l1_drain, evict* · *breaks: use-after-
  free on the meta read* · **GenMC** (cds_lrc_ownership.c).

- **INV-17** — Kind disjointness: `lrc_idx(size,kind)` clamps so an idx slot
  only ever holds one kind; pop/evict/drain derive kind from idx alone via
  `lrc_kind_from_idx`. · *lrc_idx* · *breaks: a chunk block read as a
  large_va meta (or vice-versa) → wrong size, corruption* · **comment-only**
  (the clamp is a pure function; ← Stage 2 could assert it).

- **INV-18** — A block is in ≤ 1 slot at any time. The second push of a block
  requires an intervening pop (a block can't be freed twice without being
  reallocated). · *recycle_push / pop discipline + correct client alloc/free*
  · *breaks: two slots reference one block → INV-15 violated* · **GenMC**
  (cds_lrc_ownership.c, via the double-ownership detector).

- **INV-19 (§28.1, raise-only auto-tune)** — `g_lrc_lazy_interval_ns` is only
  ever raised above the 10 ms default by auto-tune, never lowered (a single
  cold-cache munmap probe can under-estimate sustained cost). · *lrc_auto_tune
  _lazy_interval* · *breaks: down-tune makes per-thread munmap pressure WORSE
  than default (the Ohtaka 1 ms regression)* · **test** (alloc_tune_report
  self-check).

- **INV-20 (§28.4, sharded stats)** — Tier-attribution counters are sharded
  (`g_lrc_stats[LRC_STATS_SHARDS]`, cache-line-aligned, per-thread shard via
  `kame_owner_id()`). A single shared atomic here caused a ~10× MT regression
  in the chunk/large tiers. `get_stats` sums shards; individual shards may go
  transiently negative under cross-thread free, the sum is correct. ·
  *stats_inc/dec_\**, kame_pool_get_stats* · *breaks: not correctness — MT
  performance collapse* · **test** (alloc_tune_report MT sweep, alloc_stats_test).

## 7. Thread lifecycle & TLS (§20 / §23)

- **INV-21** — `release_dll_chunks_for_thread` reads `next` BEFORE setting
  BIT_OWNER_EXITED on a non-empty chunk (cached-next), because once exited a
  cross-thread returner can delete the chunk. · *release_dll_chunks_for_thread*
  · *breaks: walk dereferences freed `m_dll_next`* · **comment-only**.

- **INV-22** — Hot TLS (`g_thread_freelist_ptr`, `s_tls_owner_id`,
  `s_alloc_tls_off`, the L1/stats/lazy IE-TLS pointers) is `initial-exec`;
  the library therefore **cannot be `dlopen`'d** after startup (must be
  `LD_PRELOAD`/`-l` linked or inline-compiled). · *ALLOC_TLS_IE* · *breaks:
  `dlopen` fails / static-TLS surplus exhaustion* · **comment-only**.

- **INV-23** — After `AllocThreadExitCleanup::~dtor` sets `s_alloc_tls_off`,
  any later allocation on that thread falls through to `std::malloc` (pool TLS
  is dead). Cross-thread frees of this thread's chunks still work via
  BIT_OWNER_EXITED. · *AllocThreadExitCleanup, cold_first_access* · *breaks:
  alloc through a released chunk during teardown* · **test** (alloc_stress
  2000-thread spawn/exit).

## 8. atomic_smart_ptr / LOCAL_REF_CAPACITY (shared with kamestm)

- **INV-24** — `LOCAL_REF_CAPACITY` is the allocator's guaranteed alignment
  (`sizeof(intptr_t)`/`sizeof(double)` = 8), NOT `alignas(N)`. The tagged-
  pointer scheme stores the local refcount in the low bits, which are zero
  only if the allocator honours that alignment. · *atomic_smart_ptr.h* ·
  *breaks: pointer corruption / rare crashes if low bits non-zero* · **GenMC**
  (kamestm cds_atomic_shared_ptr suite).

---

## Verification coverage summary

| subsystem | machine-checked | comment-only (Stage 2 candidates) |
|---|---|---|
| chunk-claim state machine (§13) | INV-6,7,8 (TLA+) | — |
| chunk geometry (§15) | INV-1,2,4,5 (test) | INV-3 (compile-assert only) |
| back_offset (§15/§22) | INV-12 (TLA+), INV-13 (test) | — |
| radix (§13.3/§19/§27) | INV-10,11 (test) | **INV-9 (publish/clear ordering)** |
| recycle cache (§21–§28) | INV-15,16,18 (GenMC), INV-19,20 (test) | **INV-17 (kind disjointness)** |
| thread lifecycle (§20/§23) | INV-23 (test) | **INV-21, INV-22** |
| owner-id (§12) | INV-14 (test) | — |

**Strongest Stage-2 targets** (relied-on, unchecked, high blast radius):
INV-9 (radix publish/clear RC11 ordering), INV-21 (cached-next vs cross-thread
delete — this is the class the CrossDeallocBatch flush crash fell into), and a
TLA+ model of the cross-thread free batch (`CrossDeallocBatch::flush`) which is
currently entirely comment-only.
