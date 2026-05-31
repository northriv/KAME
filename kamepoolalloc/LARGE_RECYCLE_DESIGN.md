# Large-allocation recycle cache — §25 global log-slot

Covers the recycle layer in front of the two large tiers:

- **§15 dedicated chunk** (256 KiB – 4 MiB): one N-unit chunk per alloc, claimed
  from a 32-MiB pool region. `kind = LRC_CHUNK`.
- **§19 mmap tier** (4 MiB – 32 MiB): one 32-MiB-aligned mmap per alloc,
  radix-registered. `kind = LRC_MMAP`.

The recycle cache holds *freed-but-warm* blocks so a reuse skips the cold cost
(chunk: claim + `madvise`; mmap: `mmap` + first-touch zero) and, above all, the
**re-fault / re-zero** of the pages.

## History

| phase | cache | issue it hit |
|---|---|---|
| §21/§22 | per-thread LIFO, 8 entries / 64 MiB | working-set cliff: a rotating set > 8 blocks misses every cycle |
| §23 | same LIFO, moved to **IE-TLS** | fixed `__tls_get_addr` (35 % CPU on a 64 KiB hot loop) but kept the 8-entry cliff |
| **§25** | **global log-slot** (this doc) | no TLS at all, **no cliff**, lock-free, byte-capped |

## Structure

One atomic pointer per **log2-spaced size index** over
`[ALLOC_MIN_CHUNK_SIZE (256 KiB), ALLOC_MIN_MMAP_SIZE (32 MiB)]`:

```
g_lrc_slot[LRC_N + 1]      atomic<char*>   one cached block (base) per index, or null
g_lrc_bytes                atomic<int64>   total cached bytes (sloppy)
g_lrc_cap                  atomic<int64>   byte cap, default ~1 GiB (tunable, dynamic)
idx(S) = N·log2(S/LO)/log2(HI/LO)          integer (CLZ + 8-bit mantissa interp; no libm)
```

- **push (free)** — first EMPTY slot in the symmetric **±10 % band**
  `[idx(S)−DELTA, idx(S)+DELTA]`, one **weak** CAS per slot. Over cap / band full
  ⇒ return false, caller releases.
- **pop (alloc)** — scan the band upward, take with one **weak** CAS per slot,
  then **VERIFY the owned block's real size ≥ need** (read from the kind's meta
  *after* the CAS-take — own-then-read, never a peek ⇒ no use-after-free). Too
  small ⇒ one put-back attempt, else release. Band miss ⇒ fresh alloc.
- **kind by slot range** — bands are clamped at the chunk/mmap boundary slot
  (`BND = idx(4 MiB) = 571`), so chunk blocks live in `[0, BND]` and mmap blocks
  in `(BND, N]`; they never collide. The caller's `kind` identifies a taken
  block; its size is read from the right meta (chunk: `DEDICATED_SIZE` field;
  mmap: `LargeAllocMeta::mmap_size`).
- **±10 % band = a FIXED slot delta** in log space (`DELTA = round(N·log2(1.1)/
  log2(128)) = 20`); the boundary slot is constant — both hardcoded so the hot
  path is ONE integer `idx()` call, not three. (`static_assert` guards the
  hardcoded `BND`/`DELTA` against param drift.)

## Properties

- **Livelock-free.** Each op is a *bounded* band scan (≤ ~2·DELTA slots), ONE
  weak CAS per slot, no retry-until-success loop, deterministic fallback
  (pop → fresh alloc, push → release). Completes in O(band) steps regardless of
  contention. A 128-thread narrow-band stress completes (proto `test_livelock`).
- **weak CAS** (not strong): a single LL/SC attempt on ARM (no inner retry);
  a spurious failure just skips the slot (rare missed reuse, never wrong).
- **No working-set cliff** for the common ping-pong (free-one/alloc-one) pattern
  at *any* depth — only the in-flight freed block needs a slot. A *batch*
  (alloc-K-then-free-K) of the same size spreads across the band's ~2·DELTA
  slots, so its cliff is at ~40 (vs §22's 8); widen DELTA to raise it.
- **No TLS** ⇒ no `__tls_get_addr` on the path AND no per-thread drain.
- **Byte-capped** (sloppy single atomic, default ~1 GiB) — the RSS bound. There
  is intentionally **no periodic decay reclaim** yet (deemed premature);
  cached blocks are released only on reuse / cap-refusal / process exit.

## Why this shape (measured)

A touch-included benchmark (see proto + git history) showed the large-reuse
regime is **fault/re-zero-dominated**, not syscall-dominated: `mmap+munmap` is
~380 ns flat, but a cold first-touch zero is ~50 ns/KiB (~20 GB/s). Warm-resident
reuse avoids the re-zero (~100× faster page-touch). Hence:

- cache as high as the **RSS cap** allows (HI = 32 MiB), not a syscall crossover;
- for **real (touched) workloads** all allocators converge to memory bandwidth
  (work-bound) — the cache's value is the **no-cliff + RSS bound + latency-spike
  avoidance**, not raw small-op throughput.

Hot micro-bench (no touch, after the integer-log2 fix): 64 KiB 185, 1 MiB 164,
8 MiB 135, 16 MiB 136 M ops/s — matches/beats the §22/§23 LIFO and is flat
across the range (no cliff).

## Prototype

`tests/large_recycle_proto.cpp` is a **standalone** validator (raw-mmap backing,
not built by CMake). It carries the unit tests (`test_basic` / `test_cap` /
`test_mt` / `test_livelock`) and a touch bench. Build & run:

```
clang++ -O3 -DNDEBUG -std=gnu++17 -pthread tests/large_recycle_proto.cpp -o /tmp/lrproto
/tmp/lrproto test          # correctness + livelock
/tmp/lrproto bench 1048576 200000 fifo:16 page
```

## Tunables / drift guards

- `g_lrc_cap` — byte cap (set at runtime via the atomic; a C API can wrap it).
- `LRC_N`, `LRC_LO`, `LRC_HI` — index resolution / domain.
- `BND`, `DELTA` in `lrc_band` are precomputed for `ALLOC_MAX_CHUNK_UNITS==16`,
  `LRC_N==1000` and guarded by `static_assert` — recompute if those change.
