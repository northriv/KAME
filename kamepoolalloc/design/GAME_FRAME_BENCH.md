# Game-style frame loop (`bench_frame`) — first results (2026-10-12)

Started on the MacBook Air M3 to ask what a game developer would ask of
kamepoolalloc: does it ever make a frame late, how fast is a frame, and how
much memory does it hold.  Append results from other machines here.

## The bench

`tests/bench/bench_frame.cpp` (method in its header and in
`tests/bench/README.md`).  A game thread, a render thread and N-2 job
workers meet at a barrier at the start and end of every frame; a streaming
thread loads 1-32 MiB assets at 256 MiB/s and the game thread unloads the
oldest past a 256 MiB budget.  Per frame the game thread replaces 2000
entries of a 100 K-entry table of small objects (16 B-4 KiB, mostly under
256 B), allocates 2000 temporaries and 1500 render commands (5 % of them
4-64 KiB), and frees the previous frame's job outputs; the render thread
frees the previous frame's commands; each worker runs 8 jobs of 200 small
and 5 medium scratch blocks, 5 % of them adding a 256 KiB-4 MiB buffer.
Every block is written when allocated, and every malloc / free on the frame
threads is timed with the hardware counter.  `bench_ab.sh --only "frame
frameRT"` runs it as an interleaved A/B; `--threads N` is the frame loop's
thread count.

## M3 (4 threads: game, render, 2 workers), median of 3

`bench_ab.sh --tags v1.1.0,v1.2.0,tree --arm mi=... --arm je=... --arm sys=
--only frame --threads 4 --reps 3`; tree = 6da368e7e (the v1.2.1
allocator), mimalloc 3.3 and jemalloc 5.3 from Homebrew:

| metric | v1.1.0 | v1.2.0 | tree | mi | je | sys |
|---|---|---|---|---|---|---|
| frame median, us | 684 | 331 | 433 | **189** | 228 | 461 |
| frame p99, us | 952 | 542 | 637 | **298** | 422 | 753 |
| frame p99.9, us | 1791 | 596 | 825 | **409** | 614 | 946 |
| frame max, us | 2011 | 700 | 1549 | **468** | 819 | 4628 |
| slowest call, us | 224 | 233 | 212 | 203 | 458 | 270 |
| calls >= 100 us | 11 | 7 | **2** | 4 | 22 | 8 |
| peak RSS, MiB | 1366 | 1114 | 1115 | 465 | **404** | 646 |

- **Frame time**: mimalloc's frames are 2.3x faster than the tree's (189 vs
  433 us median), jemalloc's 1.9x; kame is level with the system allocator.
- **v1.2.0 is 30 % faster than the tree** on this workload (331 vs 433 us),
  v1.1.0 1.6x slower.  See below.
- **Tails** are noisy on this laptop (ranges over 3 runs: the tree's frame
  max 965-4897 us, mimalloc's 439-2166).  The tree has the fewest calls of
  100 us or more, but its worst frame is longer than mimalloc's because its
  frames are longer: the hitch story does not stand until the frame time
  does.  Measure tails on a quiet machine (M5, Ohtaka).
- **Memory**: the tree holds 2.4x mimalloc's peak RSS.  Its large-block
  recycle cache held 309 MiB at the end of one run (`kame_cache_mib`);
  capping it at 256 MiB (`--kame-cache-mib 256`) cut RSS by only ~60 MiB and
  added slow calls.  Part of the rest may be macOS accounting (kame releases
  chunk pages with MADV_FREE, which stays in RSS until memory pressure), but
  Ohtaka's rptest showed 1.5x on Linux too.
- `--rt` changes nothing here: the streamed assets fit the recycle cache, so
  no free unmaps (drain at most 4-7 us).

## Where the tree's frame time goes

The game thread is the frame's critical path.  Its allocator calls averaged
40 ns with kame and 22 ns with mimalloc (steady_clock timing), ~170 us per
frame over its ~9600 calls — most of the gap.  `sample` of the game thread
(tree, 4 threads): ~14 % of its time in `kame_malloc_cold` →
`PoolAllocator<32|64|256, true, false>::fl_miss`, the §fl-scan miss; the
free side ~3 %.

Random frees over a large live set (the entity churn, and the render
thread's cross-thread frees returning command slots) leave a few free slots
in each of many chunks.  A miss walks the thread's chunk list (up to
FL_SCAN_MAX = 64 per miss, from a rotating cursor) for a chunk whose
freelist is non-empty, takes it, and points the fast path there; the few
entries it holds run out quickly and the walk repeats, each step a cold
chunk header.  v1.2.0's §fl-avail kept the chunks with entries in a
per-thread list, so its misses were cheap — hence its 30 % — but paid for
the list on every free (larson 0.7x, FS=false loop 0.9x), which is why
v1.2.1 went back to the bounded walk.  mimalloc takes full pages out of its
page queue and puts them back on a free, so its search never pays for
pages without room.

## Next

1. The same on the M5 Ultra (quiet; tails) and on Ohtaka pinned to one NUMA
   node (Zen 2 like PS5 / Xbox Series, no NUMA like a console), with
   `--only "frame frameRT" --threads "8 16"`.
2. A miss as cheap as v1.2.0's at v1.2.1's free cost.  One direction: the
   free path already tests the FL_UNTRACKED marker (one tbnz) when it frees
   onto a freelist a scan confirmed empty; on that transition, push the
   chunk onto a bounded per-thread stack of chunks with entries, and let a
   miss pop it before walking.  Frees onto non-empty freelists pay nothing
   new.  Check against larson / bench_loop / replace1 (the v1.2.0
   regressions) as well as this bench.
3. Memory: what holds the RSS beyond the live set (chunk retention under
   random frees, the recycle caches), on Linux where RSS is eager.
