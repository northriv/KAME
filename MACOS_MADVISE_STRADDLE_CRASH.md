# macOS 16 KiB-page madvise straddle — crash analysis & fix (RESOLVED)

**Status: FIXED** in `f0d6a487` ("kamepoolalloc: fix macOS 16 KiB-page madvise
straddle zeroing a neighbour chunk header"). Confirmed on Apple M3 (arm64,
16 KiB pages) with a deterministic standalone repro (below) and on Linux as a
regression gate. This doc records the analysis so the separately-reported
**`CrossDeallocBatch::flush` crash can be closed on the same basis** — it is the
same zeroing, **not** a CrossDeallocBatch bug, and needs no change in
§22–§27.

> Supersedes an earlier draft of this doc that hypothesised a double-free /
> cross-dealloc UAF. **That hypothesis was wrong and is ruled out** — see
> "Why it is not a double-free" below.

---

## TL;DR

A rare `EXC_BAD_ACCESS` on macOS arm64 at thread/process exit:

```
PoolAllocator<304u,true,true>::release_dll_chunks_for_thread()  allocator.cpp:2321
  -> c->~PoolAllocator()        // VIRTUAL dtor dispatched through a NULL vtable
  -> AllocThreadExitCleanup::~  -> dyld finalizeList -> exit
```

`deallocate_chunk()`'s reclaim used
`madvise(chunk_base + ALLOC_PAGE_SIZE, chunk_size - ALLOC_PAGE_SIZE, MADV_FREE)`
to "skip the first (header) page". But §15 places each chunk's `K_MAX` (4 KiB)
header in the **4 KiB *below*** its 256 KiB unit boundary, with the slot region
starting **at** the boundary. On a target whose page size **exceeds K_MAX**
(macOS arm64 = **16 KiB**) the range ends are page-UNALIGNED, and **XNU rounds
advice ranges OUTWARD** (truncate start / round-up end) — so a chunk's
`MADV_FREE` **bleeds into the adjacent higher chunk's header page** (they share
one 16 KiB page: 12 KiB of the lower chunk's slot tail + 4 KiB of the higher
chunk's header). That zeroes a **LIVE** neighbour's embedded `PoolAllocator`
object — both its **vtable** and its **m_flags**.

That single zeroing explains the whole crash:
- `m_flags == 0` ⇒ the exit walk reads `MASK_CNT == 0` ⇒ enters the
  `newv == 0` (empty-chunk) **release** branch;
- `vtable == 0` ⇒ `c->~PoolAllocator()` (virtual dtor) dispatches through a null
  vtable ⇒ `address=0x0`.

- **macOS-only**: Linux has 4 KiB pages (`PAGE == K_MAX == 4 KiB`), all ends
  already aligned → no straddle.
- **Latent until `cbd0462c`**: that made reclaim-on-exit the default, so these
  madvises became routine.
- **Rare**: macOS `MADV_FREE` zeroing is **lazy** (the kernel only reclaims the
  freed page under memory pressure), so the live neighbour's header is zeroed —
  and the crash fires — only when the straddled page happens to be reclaimed
  before the next dispatch touches it.

## Why it is **not** a double-free

`deallocate_chunk` only clears `chunk_header[0..15]` (palloc / size_info); it
**never** touches the embedded `PoolAllocator` object at `chunk_base + 0x40`
(vtable + m_flags). The crash dump showed that **entire +0x40 object zeroed** —
which **only a page reclaim** can do. A double-free returns a slot to a bitmap;
it cannot zero a live chunk's header object. So the signature
(`vtable == 0 && m_flags == 0`) is uniquely a page-reclaim straddle.

## Same cause behind the `CrossDeallocBatch::flush` crash

The other reported crash —
`PoolAllocator<48u,true,true>::allocate_chunk_path` → inlined
`CrossDeallocBatch::flush` → `entries[k].chunk->batch_return_to_bitmap(...)` with
`address=0x18` (vtable+0x18, vtable ptr == 0) on a `CoreImageAnalytics` GCD
worker — is the **same zeroing**: a still-referenced chunk whose header page was
straddled by a neighbour's madvise → null vtable on the next virtual dispatch.
`CrossDeallocBatch` (§22–§27) is **correct**; no change needed there. The single
`deallocate_chunk` madvise fix closes both crash sites.

## The fix (`f0d6a487`)

Anchor the madvise at the **slot-region start** (`chunk_base + ALLOC_CHUNK_K_MAX`
= the unit boundary, already page-aligned by construction) and round the range
**INWARD**, so it can never cover **any** chunk's header page. Header pages stay
fully resident (this also strengthens the prior "a concurrent lookup never reads
an madvise-zeroed palloc" guarantee).

- **Linux** (`PAGE == K_MAX == 4 KiB`): the reclaimed byte range is **identical**
  to before → pure regression gate.
- **macOS**: now reclaims the lower slot pages the old unaligned start skipped,
  and leaves resident the single top page shared with the next chunk's header.

## Verification

**Linux** (range identical there): `ctest` 2/2, `alloc_stress_test`
(2000 threads, `sentinel_fails=0 diff=0`), `alloc_bucket34_repro` clean.

**macOS arm64 / 16 KiB pages** — deterministic standalone repro
`tests/alloc_madvise_straddle_repro.cpp` (forces the otherwise-lazy `MADV_FREE`
reclaim with memory pressure, then checks whether any live neighbour chunk's
header vtable was zeroed):

| allocator build | result |
|---|---|
| **pre-fix** (`0c776ea0`) | FAIL — **72 / 75** live chunk headers zeroed (3/3 runs; sometimes SIGSEGV) |
| **post-fix** (`f0d6a487`) | PASS — 0 / 75 zeroed, all headers intact |

Build/run (standalone dylib, no kame rebuild needed):
```bash
cd kamepoolalloc
clang++ -std=c++17 -O2 -DKAMEPOOLALLOC_DYLIB -dynamiclib allocator.cpp \
  -o /tmp/libkpa.dylib -install_name @rpath/libkpa.dylib
clang++ -std=c++17 -O2 tests/alloc_madvise_straddle_repro.cpp \
  -o /tmp/straddle -L/tmp -lkpa -Wl,-rpath,/tmp -lpthread
/tmp/straddle 800000 4096      # PASS post-fix; FAIL/SIGSEGV pre-fix
```
(`alloc_madvise_straddle_repro` self-skips on targets where `page <= K_MAX`,
e.g. Linux/4 KiB — no straddle possible there.)

**Remaining in-app check (optional):** rebuild kame.app on M3 and repeat
launch→use→quit several times; the rare exit crash should be gone. (A reliable
way to capture any future allocator crash without interactive lldb is to launch
under `lldb --batch -K <dumpscript> -o run <binary>` so an on-crash hook dumps
the faulting instruction + registers + each register's target memory to a file.)

## Key locations

| What | Location |
|---|---|
| the fix (madvise range) | `allocator.cpp` `deallocate_chunk()` (~`:2424`) |
| crash site A (exit walk) | `allocator.cpp:2321` `release_dll_chunks_for_thread` → `c->~PoolAllocator()` |
| crash site B (batch flush) | `allocator.cpp:1927-1928` `allocate_chunk_path` → `CrossDeallocBatch::flush` |
| virtual dtor (why null-vtable faults) | `allocator_prv.h:486` `~PoolAllocatorBase()` |
| §15 forward-shift layout (header below unit boundary) | `allocator_prv.h` (`ALLOC_CHUNK_K_MAX`, §15 notes) |
| reclaim-on-exit default that surfaced it | `cbd0462c` / `s_thread_exit_reclaim` (`allocator.cpp:2335`) |
| standalone macOS repro / regression test | `tests/alloc_madvise_straddle_repro.cpp` |
