# kamepoolalloc — Subsystem & §-chapter map

The allocator is developed as numbered design phases (`§N` tags scattered
through `allocator.cpp` / `allocator_prv.h`).  This file groups the 28-odd
`§`-tags into the ~9 real subsystems, points at where each lives, and links
the detailed design docs.  Use it to navigate: "I need to change the recycle
cache" → §21–§28 → `allocator.cpp` cache section + `LARGE_RECYCLE_DESIGN.md`
+ INVARIANTS §6.

For the invariant blast-radius map see [`INVARIANTS.md`](INVARIANTS.md).

---

## Subsystem table

| Subsystem | §-tags | Primary code | Design doc | Invariants |
|---|---|---|---|---|
| **Region & 2-level radix** | §13.2, §13.3, §19(radix) | `allocator.cpp` radix_insert/lookup/clear, `s_radix_l1`, `RadixL2Node`; `allocator_prv.h` RegionMeta | — (in-code) | INV-9,10,11 |
| **Chunk claim & bitmap state machine** | §13 | `claim_chunk`, `batch_clear_impl`, `m_flags_packed` (BIT_OWNED/MASK_CNT) | [`../tests/CHUNK_CLAIM_TLA_NOTES.md`](../tests/CHUNK_CLAIM_TLA_NOTES.md) + `../tests/tlaplus/Chunk*.tla` | INV-6,7,8 |
| **Chunk geometry (forward-shift)** | §15 | `allocate_dedicated_chunk`, `deallocate_chunk`, header offset macros, `mempool()` | — (in-code §15 comments) | INV-1,2,3 |
| **back_offset & O(1) chunk-from-slot** | §15, §22 | `s_back_offset[]`, `resolve_chunk_from_slot`, dedicated free fast path | CHUNK_CLAIM_TLA_NOTES | INV-12,13 |
| **Bucket pool & per-thread DLL** | §12, §12.3, §24 | `PoolAllocator<ALIGN,FS>`, `m_freelist_head[]`, `s_tls.dll_*`, `scan_dll_freelist` | — (in-code) | INV-14,21 |
| **Full-usable m_sizes / aligned alloc** | §16, §17 | `create()` count math, `m_sizes`/`m_align_shift`, `bucket_for_aligned`, `kBucketAlign[]` | — (in-code §16/§17) | INV-3,4 |
| **Dedicated chunks (32 KiB–4 MiB)** | §15, §22 | `allocate_dedicated_chunk`, `recycle_release_chunk` | — | INV-1,4,13 |
| **Large mmap / huge (4 MiB+)** | §19, §27 | `allocate_large_va`, `deallocate_large_va`, `large_va_raw_map/unmap`, `LargeAllocMeta` | — (in-code §19/§27) | INV-10 |
| **Recycle cache L1/L2 (K-line)** | §21–§26, §28, §28.1–§28.4 | `g_lrc[]`, `lrc_idx`, `global_push/pop_fit`, `l1_*`, `recycle_push/pop_fit`, `lrc_lazy_mmap_one`, sharded stats | [`../LARGE_RECYCLE_DESIGN.md`](../LARGE_RECYCLE_DESIGN.md) + [`../tests/cds/`](../tests/cds/) | INV-15..20 |
| **Thread lifecycle & TLS** | §20, §23 | `AllocThreadExitCleanup`, `release_dll_chunks_for_thread`, `l1_drain`, `ALLOC_TLS_IE` | — (in-code) | INV-21,22,23 |
| **OOM / C-API / observability** | §18, §28.2, §28.4 | `operator new` handler loop, `kame_pool_*` C API, `kame_pool_get_stats` v2, `g_lrc_stats` | `kame_pool.h` | INV-20 |
| **macOS 16 KiB-page straddle** | §28 (geometry) | `deallocate_chunk` madvise, `create()` slot page-bounding | [`../MACOS_MADVISE_STRADDLE_CRASH.md`](../MACOS_MADVISE_STRADDLE_CRASH.md) | INV-4,5 |

> Note: the three pre-existing design docs (`LARGE_RECYCLE_DESIGN.md`,
> `MACOS_MADVISE_STRADDLE_CRASH.md`, and `../tests/CHUNK_CLAIM_TLA_NOTES.md`)
> currently live at `kamepoolalloc/` and `kamepoolalloc/tests/` respectively;
> the links above are relative to `kamepoolalloc/design/`.  They were left in
> place (not moved into `design/`) to avoid breaking existing references from
> README and the tests.

## §-chapter chronology (what each phase added)

Phase numbering reflects the development order, not the runtime tier order.
Recovered from `git log kamepoolalloc/allocator.cpp` and the in-code tags.

| § | Added |
|---|---|
| §12 / §12.3 | Per-thread chunk DLL; compact local-id freelists (`m_freelist_head[]`). |
| §13 / §13.2 / §13.3 | 2-level radix O(1) pointer→region; embedded RegionMeta; push-only region list (retired the O(N) region scan). |
| §14 / §14B / §14C | Stats walk; NUMA-aware region lists (`mbind`, per-node DLL heads). |
| §15 | Forward-shift chunk geometry (`chunk_base = unit_boundary − K_MAX`) — fixed the dedicated-chunk free SEGV. |
| §16 | Full-usable `m_sizes[]` mode (ALIGN ≥ 1024, FS=false) — 0 % page round-up. |
| §17 | Aligned allocation served from the pool (`posix_memalign`/`aligned_alloc` ≤ 4 KiB). |
| §18 | Standards-conformant OOM (`std::new_handler` loop → `bad_alloc`; nothrow/C-API → nullptr+ENOMEM). |
| §19 | Large mmap tier (4–32 MiB): one 32-MiB-aligned mmap per alloc, radix-registered, munmap on free. |
| §20 | Cross-thread-free vptr-after-release UB fix (cache dll fields pre-call). |
| §21 / §22 | Per-thread recycle cache; shared across both large tiers (CHUNK + MMAP). |
| §23 | IE-TLS the recycle cache + `s_alloc_tls_off` hot slots (killed `__tls_get_addr` 35 % CPU). |
| §24 | `slow_allocate` scans DLL freelists across chunks (`scan_dll_freelist`). |
| §25 / §25.1 | Global lock-free log-slot cache (replaces the §23 per-thread LIFO). |
| §26 / §26.1 | Per-thread L1 in front of the global L2; index-cut per-thread bound. |
| §27 | Serve > 32 MiB from the pool (multi-region mmap + cache bypass for the huge class). |
| §28 | K-line cache (K-major, 1:1 size rounding); slot page-bounding; two-phase cap-evict. |
| §28.1 | Amortised lazy drain of the MMAP-tier cache (per-thread 10 ms tick). |
| §28.2 | `kame_pool_get_stats` v2 tier-attribution fields. |
| §28.3 | Auto-tune the lazy interval from a startup munmap probe (raise-only). |
| §28.4 | Shard the §28.2 counters (fix the MT cache-line-bounce regression). |

## How to change this allocator safely (the AI-assisted-edit recipe)

1. Identify the subsystem from the table above; read its §-tags in code.
2. `grep` the fields you'll touch in [`INVARIANTS.md`](INVARIANTS.md); note
   the **breaks-if-violated** and **verification** columns.
3. If you touch a TLA+/GenMC-checked invariant, re-run that model
   (`tests/cds/Makefile`, `tests/tlaplus/`) — and update the spec in the
   **same change** if the protocol changed.
4. If you touch a **comment-only** invariant, prefer to add a check/test;
   these are the unguarded ones (the CrossDeallocBatch flush crash lived in a
   comment-only area — INV-21).
5. Run the cmake test suite (`ctest`) + `alloc_tune_report` on the target.
6. Linux is the fast regression gate; macOS arm64 (16 KiB pages) is the
   geometry-edge gate (INV-4, INV-5) — `alloc_madvise_straddle_repro`.
