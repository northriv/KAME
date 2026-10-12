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

## Thread lists and fit lists (2026-10-12, branch claude/thread-freelist)

The miss went further than a list of chunks with entries.  The per-bucket
shortcut into one chunk's freelist is gone: a thread's own frees go onto
its list for the bucket (`KameTlsPage::m_slots[b]`), whatever the chunk,
and a malloc pops it, so a miss means the thread holds no free slot of the
bucket at all and there is nothing to look for (§thread-list).  The room a
variable-size chunk keeps when it fails a request goes on per-thread fit
lists, by the largest of the tier's sizes it still fits (§fit), instead of
v1.2.1's §walk over up to 64 chunks.  No walk over chunks is left on the
allocation path.  (A first version kept the chunk freelists and queued
the ones with entries per bucket, also walk-free; the thread lists beat it
by 1.2-1.5x on churn, having no freelists to queue.)

`bench_ab.sh --tags tree --arm tc=... --arm fit=... --arm mi=... --only
"loop larson xthread frame xlat" --threads 4 --reps 3`, median of 3; tree
= 9ccf1167b (v1.2.1), tc = thread lists with that room cut into freelist
slots (the first way tried), fit = thread lists + fit lists:

| metric | tree | tc | fit | mi |
|---|---|---|---|---|
| frame fps | 2064 | 3361 | 3206 | **4694** |
| frame median, us | 443 | 264 | 278 | **200** |
| frame p99, us | 657 | 635 | 613 | **385** |
| frame p99.9, us | 1030 | 987 | 981 | **651** |
| slowest call, us | 864 | 584 | **429** | 1752 |
| peak RSS, MiB | 1114 | 998 | 999 | **483** |
| loop 64 B / 1 KiB / 16 KiB, M ops/s | 702 / 525 / 536 | 747 / 533 / 576 | 747 / 532 / 576 | 512 / 482 / 207 |
| xthread w4 1 KiB, M frees/s | 38.0 | 41.8 | 42.4 | **97.9** |
| larson t4, M ops/s | 110 | 128 | 125 | **147** |
| xlat 1 KiB x 200 K, M/s (p99.9 ns) | 6.77 (375) | 5.99 (500) | 6.70 (417) | 10.3 (1459) |
| xlat 3000 B x 200 K, M/s (p99.9 ns) | 5.40 (500) | 4.10 (708) | 5.33 (500) | 6.26 (2584) |

- **Frame time** 0.63x the tree's (278 vs 443 us median, fps 1.55x);
  mimalloc's frames are still 1.4x faster.
- **Tails**: this run followed an hour of benchmarks on the fanless Air, and
  every arm had multi-ms frames in some rep (frame max: tree up to 41 ms,
  fit 40 ms, mi 8.6 ms).  An earlier run on a cooler machine gave tc a frame
  p99 / p99.9 / max of 280-294 / 358-409 / 475-487 us in its two clean
  reps, against mimalloc's 375-454 / 511-660 / 2501-3025 us -- the hitch
  story the first results wanted, but it has to stand on the quiet machines.
- **Cutting the room into slots** (tc) loses on producer/consumer work:
  slots cut out of cross-thread holes no longer merge with the frees around
  them (xlat 0.88x and 0.76x the tree).  Fit keeps the room in the bitmap
  and is level with the tree there, at a frame median ~5 % above tc's.
- **Single-thread churn** (`tests/bench/micro`, interleaved, median of 5;
  walk8 = the tree's design with the fl-scan capped at 8 chunks, its best
  setting): replace1 M ops/s walk8 / fit, 400-1000 B 214 / 289, 16-4096 B
  at 100 K live 55.7 / 86.0, 1500-3000 B 183 / 280; heavy (allocation
  ahead of frees) 193 / 100 ns per allocation.  Chunks live at the end
  match the tree's (511 / 511 at 100 K live); cutting the room into slots
  held 4 % fewer.

Later the same day: without the word cache, an FS=true miss claimed one
slot per bitmap CAS through the cold path, which a producer whose consumer
frees pays on every allocation -- an SPSC ring of 64 B blocks (32 in
flight) fell to 16 M ops/s from master's 28.  A miss now claims the whole
word in that CAS, and the thread's list takes its slots in one splice: 50
M ops/s.  The set again on the same Air, by then hot (`--only "frame larson
xmalloc xthread"`, median of 3):

| metric | tree | thread lists | mi |
|---|---|---|---|
| frame fps | 2040 | 3128 | **4293** |
| frame median, us | 450 | 276 | **213** |
| frame p99 / p99.9, us | 699 / 876 | 672 / 865 | **519** / 1030 |
| peak RSS, MiB | 1096 | 969 | **464** |
| xmalloc w4 64 B, M frees/s | 184 | **207** | 206 |
| larson t4, M ops/s | 99.4 | 119 | **134** |
| xthread w4 1 KiB, M frees/s | 31.0 | 35.3 | **80.7** |

## Next

1. The same on the M5 Ultra (quiet; tails) and on Ohtaka pinned to one NUMA
   node (Zen 2 like PS5 / Xbox Series, no NUMA like a console), with
   `--only "frame frameRT" --threads "8 16"`.
2. Done on the branch: the miss without a walk (above).  Remaining loops on
   the allocation path are within one chunk (bitmap scans, <= `m_count`
   words), amortised (the revival stack), or over orphaned groups (the
   FULL sweep, right before an mmap).
3. Memory: what holds the RSS beyond the live set (chunk retention under
   random frees, the recycle caches), on Linux where RSS is eager; and the
   xthread gap to mimalloc (2.3x).
