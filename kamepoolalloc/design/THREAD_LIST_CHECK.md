# Thread lists and fit lists (`5fc34e052`) — checks for the M5 Ultra and Linux

`5fc34e052` (master `04d904ccf` with its bench commit) takes every walk over
chunks off the allocation path.  The M3 results are in its commit message and
in `GAME_FRAME_BENCH.md`; the bounds in `RT_READINESS.md`.  This note lists
what the MacBook Air could not settle and how to run it elsewhere.  Append
each run below as a section, with provenance (`bench_ab.sh` writes it).

The v1.2.1 allocator to compare against is master `45f1d07f9`'s
`kamepoolalloc/` (no tag was cut).

## What changed

- Owner frees go onto the freeing thread's list for the slot's bucket
  (`KameTlsPage::m_slots[b]`), whichever chunk holds the slot; mallocs pop
  it.  A miss means the thread holds no free slot of the bucket, and goes to
  bitmap room.  The per-bucket shortcut into one chunk's freelist, §fl-scan
  (`FL_UNTRACKED`, `fl_scan`, `fl_retarget`) and §walk are gone.  The bucket
  comes from the borrow prefix, `m_base_bucket`, or `m_sizes`, which now
  records `(N << 8) | bucket` in the full tier.  Thread exit hands the lists
  back to their chunks' freelists (`tl_return_to_chunk`) for the exit walk.
- Fit lists (§fit): bitmap room a variable-size chunk keeps when it cannot
  serve a request, filed by the largest tier size it still fits; at most
  `FIT_TRIES` (4) chunks tried per request.
- An FS=true miss claims a whole bitmap word in one CAS and chains its slots
  (≤ 63 stores) for the thread's list.  The word cache's mask, the FIFO and
  the STASH experiments are removed; `m_sizes` is no longer in a union with
  the ring's counters, so the chunk object is 4 B smaller on ILP32 (LP64:
  320 B either way, the bitmap still at +384).

## What to look at

1. **ILP32.**  The union removal changes the 32-bit layout, and the claim's
   chain loop runs on a 32-bit `FUINT` in the `KAME_FORCE_UINT32_BITMAP`
   fallback (`__builtin_ctzll` on the word widened to 64 bits).
2. **No-DCAS.**  The new code adds no 64-bit atomic; phase 2 of
   `tools/audit/check_no_dcas.sh` (i486 objects export no `__atomic_*_8`)
   runs only where gcc-multilib is installed.
3. **Thread exit.**  The lists' hand-back is new: the thread-exit soaks.
4. **Zen 2.**  The word cache avoided per-slot stores because of Zen 2's
   store-port plateau (the reason `8a4d26229` gave when it made the word
   cache the default); the claim now writes one link per slot.  On the M3 the SPSC ring went from 28 (v1.2.1) to 50 M ops/s;
   whether Zen 2 keeps that is open.
5. **Tails.**  The Air throttles after ~30 minutes of benchmarks; frame
   tails were noisy for every arm, mimalloc included.
6. **Memory.**  Owner-freed slots stay on the thread's lists until it exits,
   as they stayed on chunk freelists before; peak RSS was 0.88x v1.2.1's on
   `bench_frame`, still ~2x mimalloc's.

## Linux

Six trees as in `POOL_FL_AVAIL_LINUX_CHECK.md` §1 (LP64, i586, i486;
release and asserts on), at master `04d904ccf`:

- `tools/audit/check_no_dcas.sh`, with gcc-multilib (3 phases).
- ctest in all six.  `transaction_wait_budget_test` misses its p99.99 bound
  under `ctest -j` on a small host; run it alone before counting it.
- Soaks, scored by exit status: `alloc_tsd_exclusivity_test`,
  `alloc_tsd_exclusivity_var_test` and `alloc_tsd_exclusivity_mid_test`
  2000 runs each on LP64, 500 on i486 and i586; `alloc_fit_test` 300;
  the `alloc_thread_exit_*` tests 200 each.

## M5 Ultra

Build the v1.2.1 arm from the archive, then run the A/B from this checkout:

    mkdir -p ~/bench-ab/arms/v121
    git -C ~/KAME archive 45f1d07f9 kamepoolalloc | tar -x -C ~/bench-ab/arms/v121
    cd ~/bench-ab/arms/v121/kamepoolalloc && cmake -B out -DCMAKE_BUILD_TYPE=Release && cmake --build out -j
    cd ~/KAME && kamepoolalloc/tests/bench/bench_ab.sh --tags tree \
        --arm v121=$HOME/bench-ab/arms/v121/kamepoolalloc/out/libkamepoolalloc.dylib \
        --arm mi=/path/to/libmimalloc.dylib \
        --only "loop xthread xmalloc larson rptest xlat frame frameRT" \
        --threads "4 8" --reps 5

Then the micro benches (`tests/bench/micro/README.md`) with `./ab.py . 5
v121=... tree=...`, and STM `transaction_payload_integrity_3level_mixed_test`
from two test trees, one per allocator, interleaved, with exactly four
arguments: `3 128 128 10`, `3 128 128 0`, `3 8 128 10`.  Keep the frame loop
within the performance cores (`--threads` ≤ 10 there).

## Ohtaka (Zen 2, like PS5 / Xbox Series)

The same `bench_ab.sh` set as the M5's, under the Ohtaka rules in
`CLAUDE.md` (`srun --exclusive` on `i8cpu`, one job per measurement, median
of 5, provenance): `--threads "4 16 64" --xlat-cpu 0,1`, and the frame loop
pinned to one NUMA node (`numactl -N 0`), as a console has none.  Item 4
above is the one to watch: ring and `xmalloc-test` 64 B against v1.2.1.

## Run: M5 Ultra, 2026-10-12 (master `ac4144493`)

Machine: Apple M5 Ultra (10P + 20E, 96 GiB, Mac17,15), macOS 27.0.1
(26A434), Apple clang 21.0.0, AC power.  v1.2.1 arm built from the archive
of `45f1d07f9` as above (sha256 `15903cd5c08a8c1f`); tree at `ac4144493`
(sha256 `6d9157ba1b90e29d`); mimalloc 3.3.2.

**Correctness.**  Full-tree ctest 47/47 in Release and in `-O2 -g` with
asserts.  Thread-exit soaks (item 3), both builds, scored by exit status, 6
in parallel: `alloc_tsd_exclusivity_test` and `alloc_tsd_exclusivity_var_test`
2000 each, `alloc_fit_test` 300, `alloc_thread_exit_free_test`,
`_free_test_dynamic`, `_unarmed_test`, `_unarmed_test_dynamic` 200 each —
0 failures.  (No `alloc_tsd_exclusivity_mid_test` in the macOS tree.)

**`bench_ab.sh`** (the command above, `--threads "4 8" --reps 5`; results
`~/bench-ab/results-dhcp-029-018-20261012-222036`), median (min–max):

| bench | config | metric | tree | v121 | mi | v121 / tree | mi / tree |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1169 (1135–1172) | 1132 (1129–1137) | 1066 (1010–1070) | 0.97× | 0.91× |
| loop | 1024B | Mops ↑ | 840 (807–851) | 790 (750–806) | 775 (742–803) | 0.94× | 0.92× |
| loop | 16384B | Mops ↑ | 839 (825–884) | 837 (830–895) | 362 (359–365) | 1.00× | 0.43× |
| xthread | w4,1KiB | Mfree ↑ | 87.3 (86.6–88.9) | 80.8 (78.8–84.0) | 104 (98.9–108) | 0.93× | 1.19× |
| xmalloc | w4,64B | Mfree ↑ | 389 (385–394) | 381 (380–393) | 348 (344–351) | 0.98× | 0.89× |
| larson | t4 | Mops ↑ | 163 (163–166) | 148 (145–150) | 195 (194–195) | 0.91× | 1.20× |
| rptest | t4 | Mops_cpu ↑ | 7.61 (7.15–7.85) | 6.93 (6.75–7.38) | 7.24 (6.92–7.75) | 0.91× | 0.95× |
| frame | t4 | fps ↑ | 4945 (4933–5011) | 2467 (2454–2475) | 5339 (5253–5369) | 0.50× | 1.08× |
| frame | t4 | frame_p99_us ↓ | 240 (230–240) | 426 (418–430) | 240 (232–256) | 1.78× | 1.00× |
| frame | t4 | frame_max_us ↓ | 358 (325–384) | 569 (562–587) | 404 (340–1052) | 1.59× | 1.13× |
| frame | t4 | call_max_ns ↓ | 56534 (44867–106010) | 76742 (75325–102736) | 177171 (134903–849784) | 1.36× | 3.13× |
| frame | t4 | rss_mib ↓ | 931 (930–931) | 1076 (1075–1076) | 471 (462–490) | 1.16× | 0.51× |
| frameRT | t4 | fps ↑ | 4949 (4920–5026) | 2466 (2460–2471) | 5337 (5314–5454) | 0.50× | 1.08× |
| frameRT | t4 | frame_max_us ↓ | 344 (304–391) | 554 (513–683) | 412 (354–1227) | 1.61× | 1.20× |
| xthread | w8,1KiB | Mfree ↑ | 129 (124–130) | 123 (119–124) | 79.7 (77.1–85.6) | 0.95× | 0.62× |
| xmalloc | w8,64B | Mfree ↑ | 501 (497–502) | 504 (504–506) | 496 (493–500) | 1.01× | 0.99× |
| larson | t8 | Mops ↑ | 251 (247–255) | 227 (226–230) | 202 (198–207) | 0.91× | 0.81× |
| rptest | t8 | Mops_cpu ↑ | 6.89 (6.49–7.23) | 6.14 (5.98–7.47) | 5.71 (5.3–6.11) | 0.89× | 0.83× |
| frame | t8 | fps ↑ | 3946 (3849–3996) | 1845 (1834–1861) | 3446 (3408–3522) | 0.47× | 0.87× |
| frame | t8 | frame_p99_us ↓ | 392 (386–405) | 627 (617–630) | 470 (464–476) | 1.60× | 1.20× |
| frame | t8 | frame_max_us ↓ | 497 (479–519) | 740 (732–843) | 1476 (1034–1746) | 1.49× | 2.97× |
| frame | t8 | call_max_ns ↓ | 111760 (96402–186771) | 121527 (74552–128487) | 1131651 (943213–1540170) | 1.09× | 10.13× |
| frame | t8 | rss_mib ↓ | 1163 (1159–1163) | 1406 (1405–1406) | 579 (562–611) | 1.21× | 0.50× |
| frameRT | t8 | fps ↑ | 3930 (3792–3991) | 1857 (1849–1858) | 3478 (3458–3492) | 0.47× | 0.88× |
| frameRT | t8 | frame_max_us ↓ | 498 (450–528) | 742 (711–928) | 1548 (1413–1666) | 1.49× | 3.11× |
| xlat | 1024Bx200000 | rate_M ↑ | 5.06 (4.89–5.33) | 4.81 (4.8–4.93) | 9.28 (9.13–9.35) | 0.95× | 1.83× |
| xlat | 1024Bx200000 | p999_ns ↓ | 292 (291–333) | 292 (292–500) | 958 (917–1000) | 1.00× | 3.28× |
| xlat | 1024Bx200000 | max_ns ↓ | 33125 (18083–51875) | 35500 (14875–39625) | 21750 (15209–68792) | 1.07× | 0.66× |
| xlat | 1024Bx1000000 | rate_M ↑ | 4.96 (4.92–5.08) | 4.84 (4.81–4.86) | 8.42 (8.24–8.46) | 0.98× | 1.70× |
| xlat | 1024Bx1000000 | max_ns ↓ | 18125 (13208–29500) | 15583 (14458–43292) | 26083 (21458–60833) | 0.86× | 1.44× |
| xlat | 3000Bx200000 | rate_M ↑ | 4.18 (4.07–4.2) | 4.15 (4.01–4.26) | 6.54 (6.5–6.58) | 0.99× | 1.57× |
| xlat | 3000Bx200000 | max_ns ↓ | 16792 (15458–36708) | 14583 (13667–65750) | 20333 (19958–21208) | 0.87× | 1.21× |

(Rows left out: frame / frameRT median, p99.9, slow_calls and the
frameRT p99 / call_max / rss rows, which follow frame's; xlat med_ns.  The
full table is the results directory's `summary.md`.)  No regression: tree
is at or above v121 everywhere; xmalloc w8 0.99× and the xlat max_ns rows
are within their spread.  Frame throughput doubles, its tails shorten 1.5–
1.9×, and RSS is 0.83–0.86× v121's (item 6: still ~2× mimalloc's).

**Micro benches** (`ab.py . 5 v121=… tree=…`, median of 5):

| config | unit | v121 | tree |
|---|---|---|---|
| replace1 5000 20000000 400 1000 | M ops/s | 166.4 | 398.3 |
| replace1 5000 20000000 8 1000 | M ops/s | 157.1 | 198.9 |
| replace1 100000 20000000 16 4096 | M ops/s | 43.2 | 120.0 |
| replace1 5000 20000000 1500 3000 | M ops/s | 80.3 | 384.9 |
| replace1 5000 20000000 16 368 | M ops/s | 286.7 | 277.4 |
| heavy 100000 1000000 4 400 1000 | ns/alloc | 1603.1 | 74.3 |
| heavy 100000 1000000 32 400 1000 | ns/alloc | 1606.4 | 63.4 |
| heavy 100000 1000000 32 32 512 | ns/alloc | 90.9 | 32.7 |
| ring 64 4 20 | M ops/s | 11.5 | 13.4 |
| ring 64 32 20 | M ops/s | 12.5 | 22.4 |
| ring 64 128 20 | M ops/s | 13.1 | 24.3 |

The one loss, replace1 16..368, is real but small.  Interleaved, 7–11
runs each (tree / v121): [16, 368) 0.975, [16, 128) 0.957, [128, 368)
0.982, [16, 64) **1.086**; unchanged at 4×10⁸ iterations, so not warm-up.
Over the live-set size at [16, 128): 200 → 0.940, 1000 → 0.944, 5000 →
0.956, 50000 → **1.057** — tree loses only while everything is L1-resident,
by about 0.2 ns (one cycle) an operation, and wins once misses count.  A
Time Profiler leaf-PC histogram (`leafpc.py`, 10⁹ iterations at [16, 128))
puts the free side equal (kame_free 4544 vs 4452 samples, its main stall the
same chunk-header load in both) and the difference in malloc: 842 vs 591
samples, on the load of the thread-list head (`ldr x8, [x10, x9, lsl #3]`,
615 samples) where v121 spends ~355 on its two loads.  A guess, not
confirmed: tree's owner free stores the head to `KameTlsPage::m_slots[b]`,
an address that waits on `ldrh` of the chunk's bucket, so the next malloc's
load of a head cell is held behind it; v121's free stored to the chunk's own
cell, whose address is known early.  It does not explain [16, 64).  If it
matters, a bucket byte in the region metadata (already loaded and hot)
would take the chunk header off the store's address.

**STM** `transaction_payload_integrity_3level_mixed_test` from two test
trees (v121 = a worktree at `45f1d07f9`), interleaved, 5 runs each, all
PASS; commits/s median (min–max):

| args | tree | v121 | tree / v121 |
|---|---|---|---|
| `3 128 128 10` | 11.82 M (11.52–12.06) | 11.72 M (11.25–11.93) | 1.01 |
| `3 128 128 0` | 269.06 M (267.01–270.13) | 271.01 M (268.32–272.48) | 0.99 |
| `3 8 128 10` | 9.42 M (8.31–9.99) | 10.50 M (10.00–10.61) | 0.90 |

The 8-thread row's 0.90 is noise: 12 more interleaved runs each give mean
9.63 M (tree) vs 9.68 M (v121), both spanning 7.9–10.6 M.

Items 1, 2 and 4 (ILP32, no-DCAS phase 2, Zen 2) need Linux / Ohtaka.
`leafpc.py` read nothing from Xcode 27's export (`tagged-backtrace` in
place of `backtrace`); fixed in the same commit.

## Since the M5 run: `f9e5cb9b3` (§tl-cap) — what to re-check

`f9e5cb9b3` bounds what a thread keeps of its own freed memory: about
4 MiB per bucket, its lists sealed into 256 KiB segments
(`KameTlsPage::m_seg`) and the oldest returned to its chunks past 15 of
them; a realtime thread returns single slots instead.  The M5 run above
predates it.  Re-check on the M5 and on Linux:

1. **The owner free.**  It now reads the head slot's second word (the
   bytes below it, a load that depends on the head) and stores two words
   into the freed slot.  M3: bench_loop 64 B 1.00x, 1 KiB 0.93x, 16 KiB
   0.93x against `727c47322`.  Re-run `loop` and the micro benches; the
   L1-resident replace1 loss at 16..368 B noted above may grow, since the
   extra load sits on the free that its guess already blames.
2. **The cold paths.**  `tl_seal` / `tl_return_segment` / `owner_return`
   and the unseal in `tc_miss`: `alloc_phase_test` (new, in ctest — 48
   tests now), the thread-exit soaks and `alloc_rt_thread_test` again; a
   build with `-DKAME_TL_SEG_BYTES=1024u -DKAME_TL_SEGS=1` seals and returns
   all the time and is the harsher one.
3. **ILP32.**  `KameTlsPage` grew by `m_seg[52]` (its packed-size
   static_assert now counts it): the i486 / i586 trees.
4. **Hitches.**  The free that returns a segment pays for it: freeing
   200 MiB of 64 B or 1 KiB blocks on the M3, the slowest free took
   31–56 µs (before: 4–39 µs).  On the M5 and Ohtaka with a probe that
   times each free of such a loop; `bench_frame` saw 39 seals in 3000
   frames.
5. **Memory.**  The phase change (one thread, 400 MiB of 64 B freed, then
   400 MiB of 1 KiB) peaked at 1053 MiB before, 631 MiB now (581 with
   `kame_pool_trim()` between; mimalloc 480) on the M3; on Linux RSS is
   eager, so the number there is the one that matters.
6. **Random-order frees.**  The cap bounds the bytes a list keeps, not the
   chunks they hold: freed in a random order, the last megabytes of frees
   lie on every chunk, and the phase change above held 1003 MiB.  Since the
   next commit, a thread about to take a new chunk gives back everything a
   bucket keeps whose list has not moved over its last two new chunks
   (`tl_reclaim_idle`; 627 MiB), and `kame_pool_trim()` gives back all of the
   calling thread's lists.  `alloc_phase_test` now frees in a random order
   too; re-check it, and the thread-exit soaks, on Linux.

## Run: Linux, 2026-10-12 (master `ac4144493`, the code of `04d904ccf`)

Cloud container, x86-64, g++ 13.3, 4 vCPU, glibc; the six trees of
`POOL_FL_AVAIL_LINUX_CHECK.md` §1 (LP64, i586, i486; release `-O3 -DNDEBUG`
and asserts on `-O2 -g -UNDEBUG`), rebuilt at `ac4144493` (only this note
differs from `04d904ccf`).  Soaks scored by exit status, 4 processes at a
time, the tests of one stage interleaved round by round, an `LD_PRELOAD`
SIGSEGV reporter on every run.

| check | result |
|---|---|
| `check_no_dcas.sh` (gcc-multilib) | **3/3 ok** — phase 1 builds the `KAME_FORCE_UINT32_BITMAP` fallback with the claim's chain loop; the i486 objects of both libraries export no `__atomic_*_8` |
| build LP64 / i586, release and asserts | clean |
| build i486, release and asserts | **`bench_frame` fails to link** (below); every other target builds (`-k`) |
| ctest LP64 release / asserts | 47/47, 47/47 |
| ctest i586 release / asserts | 45/47, 46/47: **`alloc_fit_test`** (below); release also `transaction_wait_budget_test`, 3/3 PASS alone |
| ctest i486 release / asserts | 39/40, 39/40: **`alloc_fit_test`** |
| `alloc_tsd_exclusivity_test` / `_var_test` / `_mid_test`, LP64 release | **0 / 2000** each |
| the same, i586 and i486 release | **0 / 500** each (six arms) |
| `alloc_fit_test`, LP64 release | 0 / 300 |
| `alloc_thread_exit_free_test`, `_dynamic`, `alloc_thread_exit_unarmed_test`, `_dynamic`, LP64 release | **0 / 200** each |

**Item 1 (ILP32) — two findings, neither in the allocator.**

1. *`bench_frame` does not link on i486.*  `std::atomic<std::uint64_t>
   g_loads` (`tests/bench/bench_frame.cpp:323`) lowers to
   `__atomic_fetch_add_8` and `__atomic_load_8`, which an i486 target has
   no instruction for.  Without `-k` the build stops there, so the rest of
   the i486 test tree (`alloc_fit_test` included) is never built.  The
   libraries are clean — the probe above is what guards them — and
   `check_no_dcas.sh`'s scope note allows a bench a 64-bit tally, but this
   one breaks the whole i486 test build.  A pointer-width counter (or
   excluding `bench_frame` when `ATOMIC_LLONG_LOCK_FREE != 2`) would fix it.
2. *`alloc_fit_test` fails on every ILP32 tree, deterministically (5/5
   each); LP64 passes.*  Output: "no probe landed in the fillers' chunks --
   no room was reused" for both tiers; 0 slots too short, 0 blocks
   overwritten.  The test's geometry is LP64's: "a 4-unit run at the end of
   every 64-unit bitmap word (N = 6: ten per word)".  On ILP32 `FUINT` is
   `uintptr_t`, a 32-unit word: five 160 B fillers (N = 6) leave 2 units,
   too few for the 96 B probe (N = 4), and the ALIGN 256 sizes do not map
   the same way either.  The fit lists themselves work there: a scratch copy
   taking the sizes as arguments (not committed) found reuse on ILP32 at
   other sizes, and two such pairs pass 5/5 on all four ILP32 trees,
   asserts on included, with the test's full checks:

   | ILP32 pair (filler, probe) | probes among the fillers, round 1 / 2 | short slots | overwritten |
   |---|---|---|---|
   | 180 B, 96 B (6000 / 3000) | 1499 / 3000 | 0 | 0 |
   | 1000 B, 512 B (3000 / 3000) | 2927 / 300 | 0 | 0 |

   So the test wants its sizes chosen by `sizeof(void *)` (or derived from
   the bitmap word's width), not a fix in §fit.

Item 3 (thread exit): the lists' hand-back at teardown is exercised by the
exclusivity soaks and the four thread-exit soaks above, 0 failures in all.
Items 4–6 (Zen 2, tails, memory) are performance questions for a real host.

## Run: Linux, 2026-10-12 (master `6b61c6e4a`, the §tl-cap code of `f9e5cb9b3`)

Same container and six trees as the run above, rebuilt at `6b61c6e4a`, plus
three harsh trees built with `-DKAME_TL_SEG_BYTES=1024u -DKAME_TL_SEGS=1`
(LP64 release, LP64 asserts, i586 asserts; the flags confirmed in their
CMake caches).  Items 2, 3 and 5 of the list above.

| check | result |
|---|---|
| `check_no_dcas.sh` | **3/3 ok** — §tl-cap adds no 64-bit atomic |
| builds | clean, except `bench_frame` on i486 again (still `std::atomic<std::uint64_t>`, see the run above) |
| ctest LP64 release / asserts | 48/48, 48/48 |
| ctest harsh LP64 release / asserts | **48/48, 48/48** |
| ctest i586 release / asserts, harsh i586 asserts | 45/48, 45/48, 46/48 |
| ctest i486 release / asserts | 39/41, 38/41 |

On the ILP32 trees the failures are `alloc_fit_test` (as in the run above:
its sizes are LP64's), **`alloc_phase_test`** (below) and, on three of them,
`transaction_wait_budget_test` under `ctest -j3`, 3/3 PASS alone each time.

Soaks, by exit status, 4 at a time, each stage's tests interleaved:

| tree | `alloc_phase_test` | `alloc_rt_thread_test` | `alloc_thread_exit_*` (4) | exclusivity 32 B / var / mid | `alloc_fit_test` |
|---|---|---|---|---|---|
| LP64 release | 0 / 200 | 0 / 200 | 0 / 200 each | 0 / 500 each | 0 / 500 |
| harsh LP64 release | 0 / 100 | 0 / 100 | 0 / 100 each | 0 / 100 each | — |
| harsh LP64 asserts | 0 / 100 | 0 / 100 | 0 / 100 each | 0 / 100 each | — |
| harsh i586 asserts | 100 / 100 (threshold, below) | 0 / 100 | 0 / 100 each | 0 / 100 each | — |
| i586 release | 100 / 100 (threshold) | 0 / 100 | 0 / 100 each | 0 / 100 each | — |
| i486 release | 100 / 100 (threshold) | 0 / 100 | 0 / 100 each | 0 / 100 each | — |

**`alloc_phase_test` on ILP32: the cap works; the test's threshold is
LP64's.**  Every ILP32 failure (300 runs) is the one message "the second
phase stacked on the first", with 0 blocks overwritten in all 1,200 pair
checks and no crash.  The numbers, i586 (i486 identical):

| | 64 B → 1 KiB: units after phase 1 / after free + trim / after phase 2 | 1 KiB alone (first phase of 1 KiB → 64 B) |
|---|---|---|
| LP64, `6b61c6e4a` | 262 / 20 / 370 — PASS | 327 |
| ILP32, `6b61c6e4a` | 262 / 20 / **543** — FAIL (543 × 2 > 262 × 3) | 489 |
| LP64, `ac4144493` library (before the cap) | 262 / 262 / 612 — FAIL, as it should | — |
| ILP32, `ac4144493` library | 262 / 262 / 785 — FAIL, as it should | — |

The freed phase does go (262 → 20 units, also on ILP32), and 543 ≈ 20 +
the 1 KiB phase's own cost (489–523 units on ILP32).  What differs is that
cost: a 1 KiB request has the same 1080 B slot on both
(`kame_pool_malloc_usable_size`), but on ILP32 the bitmap word holds 32
units, not 64, and the 1 KiB class packs worse into it (~1.5× the units —
the same ratio before the cap: 523 vs 350 units for the 1 KiB phase on the
`ac4144493` library).  The test's `u3 ≤ 1.5 × u1` assumes LP64's packing
("1 KiB requests round up to 1280 B slots"); measured against the second
size's own cost, or with a per-width factor, it would pass.  The harsh i586
tree is the same (262 / 3 / 526; 1 KiB alone 520).

**Item 5, memory — RSS on Linux.**  One thread, 400 MiB of the first size
painted and freed, then 400 MiB of the second painted, plain `malloc` /
`free` (the pool interposes them), VmHWM; LP64, the same binary against
each library:

| phase change | `6b61c6e4a` | before the cap (`ac4144493`) | glibc |
|---|---|---|---|
| 64 B → 1 KiB | **603 MiB** | 1005 | 552 |
| 64 B → 1 KiB, `kame_pool_trim()` / `malloc_trim()` between | **553** | 955 | 552 |
| 4 KiB → 64 B | **682** | 863 | 553 |
| 4 KiB → 64 B, trim between | **466** | 863 | 553 |

After the first phase is freed, RSS is 135 MiB (64 B) and 227 MiB (4 KiB)
on `6b61c6e4a` against 460 and 407 before; the trim takes it to 15 and 11.
The M3's 1053 → 631 (581) and 891 → 707 (587) have the same shape.

Items 1 and 4 (the owner free's cost, the slowest free) are timing on a
real host; not measured on this VM.

## Run: M5 Ultra, 2026-10-13 — §tl-cap (master `6b61c6e4a`)

Same machine and toolchain as the 2026-10-12 run.  Arms: tree at
`6b61c6e4a` (sha256 `b6ea3d464599bbf6`), pre = `727c47322`'s
`kamepoolalloc/` from the archive (sha256 `6d9157ba1b90e29d`, the tree arm
of the run above), mimalloc 3.3.2.  `23ddbbd7a` and `460730c35`
(`tl_reclaim_idle`) landed after this run and are not covered.

**Correctness (items 2, 3).**  Full-tree ctest 48/48 in Release, in `-O2
-g` with asserts, and in that build with `-DKAME_TL_SEG_BYTES=1024u
-DKAME_TL_SEGS=1`.  In all three, scored by exit status, 6 in parallel:
`alloc_tsd_exclusivity_test` and `_var_test` 2000 each, `alloc_fit_test`
300, `alloc_phase_test` 300, `alloc_rt_thread_test` 200, the four
`alloc_thread_exit_*` tests 200 each — 0 failures.

**`bench_ab.sh`** (`--tags tree --arm pre=… --arm mi=…`, the same `--only`
set, `--threads "4 8" --reps 5`; results
`~/bench-ab/results-dhcp-029-018-20261013-012520`), the rows that moved
and the headline ones, median (min–max):

| bench | config | metric | tree | pre | mi | pre / tree | mi / tree |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1139 (1133–1162) | 1171 (1137–1180) | 1071 (1013–1078) | 1.03× | 0.94× |
| loop | 1024B | Mops ↑ | 750 (717–769) | 842 (761–914) | 781 (767–794) | 1.12× | 1.04× |
| loop | 16384B | Mops ↑ | 778 (724–800) | 846 (828–876) | 361 (342–371) | 1.09× | 0.46× |
| xthread | w4,1KiB | Mfree ↑ | 84.6 (84.2–86.9) | 89.6 (83.2–89.7) | 111 (103–113) | 1.06× | 1.32× |
| xmalloc | w4,64B | Mfree ↑ | 404 (386–409) | 392 (381–405) | 346 (344–346) | 0.97× | 0.86× |
| larson | t4 | Mops ↑ | 166 (164–169) | 169 (167–170) | 199 (199–201) | 1.02× | 1.20× |
| rptest | t4 | Mops_cpu ↑ | 7.06 (6.83–7.25) | 7.4 (6.48–7.78) | 7.15 (6.73–7.28) | 1.05× | 1.01× |
| frame | t4 | fps ↑ | 5213 (5092–5270) | 5277 (5011–5296) | 5601 (5496–5680) | 1.01× | 1.07× |
| frame | t4 | frame_max_us ↓ | 311 (225–365) | 350 (232–487) | 425 (275–1384) | 1.13× | 1.37× |
| frame | t4 | rss_mib ↓ | 916 (916–931) | 916 (916–931) | 483 (464–491) | 1.00× | 0.53× |
| xthread | w8,1KiB | Mfree ↑ | 125 (119–127) | 123 (119–125) | 80.0 (76.2–85.6) | 0.98× | 0.64× |
| xmalloc | w8,64B | Mfree ↑ | 511 (506–514) | 509 (504–512) | 505 (503–507) | 1.00× | 0.99× |
| larson | t8 | Mops ↑ | 256 (254–258) | 260 (259–262) | 211 (204–213) | 1.01× | 0.82× |
| rptest | t8 | Mops_cpu ↑ | 6.48 (6.08–7.27) | 6.51 (5.83–6.98) | 5.36 (5.16–5.61) | 1.00× | 0.83× |
| frame | t8 | fps ↑ | 4159 (4128–4173) | 4176 (4138–4207) | 3692 (3553–3709) | 1.00× | 0.89× |
| frame | t8 | frame_max_us ↓ | 482 (440–503) | 432 (402–662) | 1655 (1260–1700) | 0.90× | 3.43× |
| frame | t8 | rss_mib ↓ | 1158 (1158–1162) | 1158 (1158–1162) | 564 (562–608) | 1.00× | 0.49× |
| xlat | 1024Bx200000 | rate_M ↑ | 4.98 (4.92–5.12) | 5.07 (4.94–5.2) | 9.29 (9.2–9.35) | 1.02× | 1.87× |
| xlat | 1024Bx1000000 | max_ns ↓ | 14791 (13083–16958) | 13500 (13416–14208) | 22000 (20250–33250) | 0.91× | 1.49× |

The rest — frameRT, the other frame percentiles, slow calls, the other
xlat rows — are level within their spread (`summary.md` there has them
all).  Item 1: the loss is on the single-thread loop above 64 B, 1 KiB
0.89x (now below mimalloc) and 16 KiB 0.92x, against the M3's 0.93x /
0.93x; the multi-thread benches are level.

**Micro benches** (`ab.py . 5 pre=… tree=…`, median of 5):

| config | unit | pre | tree |
|---|---|---|---|
| replace1 5000 20000000 400 1000 | M ops/s | 401.9 | 391.9 |
| replace1 5000 20000000 8 1000 | M ops/s | 200.5 | 197.9 |
| replace1 100000 20000000 16 4096 | M ops/s | 127.0 | 121.5 |
| replace1 5000 20000000 1500 3000 | M ops/s | 386.4 | 356.8 |
| replace1 5000 20000000 16 368 | M ops/s | 279.1 | 275.0 |
| heavy 100000 1000000 4 400 1000 | ns/alloc | 73.6 | 74.9 |
| heavy 100000 1000000 32 400 1000 | ns/alloc | 63.6 | 63.9 |
| heavy 100000 1000000 32 32 512 | ns/alloc | 31.6 | 31.1 |
| ring 64 4 20 | M ops/s | 13.5 | 13.3 |
| ring 64 32 20 | M ops/s | 22.7 | 22.5 |
| ring 64 128 20 | M ops/s | 23.5 | 23.2 |

The L1-resident replace1 loss at 16..368 B did not grow (0.985); the
larger FS=false sizes lost more: 1500..3000 B 0.92x, 16..4096 B 0.96x.

**Hitches (item 4).**  A probe that mallocs 200 MiB of one size and times
every free in allocation order (`clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`
around each `free`), 3 runs per arm:

| size | metric | pre | tree | mi |
|---|---|---|---|---|
| 64 B | mean | 7.3–8.3 ns | 9.6–9.8 ns | 7.3–7.4 ns |
| 64 B | p99.99 | 84–125 ns | 10.3–10.9 µs | 125 ns |
| 64 B | max | 17.8–29.3 µs | 21.6–26.3 µs | 18.1–21.7 µs |
| 64 B | frees > 1 µs / > 10 µs | 18–26 / 12–17 | 801–816 / 444–527 | 19–30 / 8–14 |
| 1 KiB | mean | 26.1–28.3 ns | 55.3–57.8 ns | 8.9–9.2 ns |
| 1 KiB | p99.99 | 250 ns | 8.4–17.1 µs | 209–250 ns |
| 1 KiB | max | 17.8–22.8 µs | 29.3–39.5 µs | 10.6–12.0 µs |
| 1 KiB | frees > 1 µs / > 10 µs | 7–14 / 1–6 | 835–840 / 11–43 | 1–7 / 1–2 |

The worst free moves little; what changes is one slow free per sealed
segment returned — 200 MiB / 256 KiB = 800, the > 1 µs count — at
roughly 8–9 µs each, which pushes p99.99 from ~100 ns to ~10 µs.  At
1 KiB the returns also double the mean free (≈ 30 ns a returned slot
amortized, against ≈ 2 ns at 64 B).

**Memory (item 5).**  One thread, 400 MiB of A-byte blocks malloc'd,
freed in allocation order, then 400 MiB of B-byte blocks; peak
`phys_footprint` (`ledger_phys_footprint_peak`):

| A → B | pre | tree | tree, `kame_pool_trim()` between | mi |
|---|---|---|---|---|
| 64 B → 1 KiB | 1053 MiB | 631 MiB | 581 MiB | 481 MiB |
| 4 KiB → 64 B | 891 MiB | 707 MiB | 588 MiB | 467 MiB |

The M3's numbers to the MiB.

## Since the Linux runs: `75369a9a3` and `10e5bd4bf` — what to re-run on ILP32

The three ILP32 findings above were in the tests, and `10e5bd4bf` fixes
them: `bench_frame`'s counter is pointer-width, `alloc_fit_test` takes its
sizes by the bitmap word's width, and `alloc_phase_test` bounds what the
second phase leaves by what the second size takes alone, not by the first
phase's units.  Reading item 2 turned up a real ILP32 bug, fixed in
`75369a9a3`: `kBucketAlign` carried LP64's 32 and 256 for buckets 8 and
16, which `ALLOC_ALIGN` makes ALIGN 128 on ILP32.  A 256-aligned request
of up to 248 B went to bucket 16 and got every other slot only 128-aligned
(`aligned_alloc`, `posix_memalign`, aligned `operator new` on POSIX), and
the ALIGN 128 template had no fit classes.  A static_assert now ties every
bucket's `kBucketAlign` to its template.  LP64's `allocator.o` is
byte-identical to before; the ILP32 side has been checked only at compile
time (macOS clang `-target i386-apple-macos10.13 -fsyntax-only`, with
static_asserts on the picks and the classes).  On the four ILP32 trees:

1. **Build without `-k`.**  The i486 trees should build whole,
   `bench_frame` included.
2. **ctest.**  `alloc_fit_test`, `alloc_phase_test` and the new
   `alloc_aligned_test` should pass.  `alloc_fit_test`'s second ILP32 tier
   is the ALIGN 128 one (256 B fillers, 128 B probes), which had no fit
   classes before `75369a9a3`: its "among the fillers" counts are §fit's
   first run there.
3. **The alignment bug at run time.**  It has only been read off the
   tables so far.  `alloc_aligned_test` against a library built at
   `460730c35` (swap `libkamepoolalloc.so` with `LD_LIBRARY_PATH`) should
   fail on ILP32 with "misaligned: alignment 256, size …", and pass
   against `75369a9a3`'s; on LP64 both pass.
4. **Soaks.**  `alloc_phase_test` and `alloc_fit_test`, 100 runs each per
   ILP32 tree.

## Run: M5 Ultra, 2026-10-13 — `tl_reclaim_idle` (master `4ae71a910`, the code of `795567059`)

Same machine and toolchain as the runs above.  Arms: tree at `4ae71a910`
(sha256 `fc5fe9777215ee47`), cap = `6b61c6e4a`'s `kamepoolalloc/` (sha256
`b6ea3d464599bbf6`, the tree arm of the §tl-cap run above), pre =
`727c47322`'s (sha256 `6d9157ba1b90e29d`, before the cap), mimalloc 3.3.2.
The code since cap: `tl_reclaim_idle` and the trim of the thread's lists
(`460730c35`), the ILP32 `kBucketAlign` fix (`75369a9a3`), test fixes.

**Correctness.**  Full-tree ctest 49/49 (with `alloc_aligned_test`) in
Release, in `-O2 -g` with asserts, and in that build with
`-DKAME_TL_SEG_BYTES=1024u -DKAME_TL_SEGS=1`.  In all three, scored by exit
status, 6 in parallel: `alloc_tsd_exclusivity_test` and `_var_test` 2000
each, `alloc_fit_test` 300, `alloc_phase_test` 300 (random-order frees
now), `alloc_rt_thread_test` 200, `alloc_aligned_test` 200, the four
`alloc_thread_exit_*` tests 200 each — 0 failures.

**`bench_ab.sh`** (`--tags tree --arm cap=… --arm pre=… --arm mi=…`, the
same `--only` set, `--threads "4 8" --reps 5`; results
`~/bench-ab/results-dhcp-029-018-20261013-103556`), median (min–max):

| bench | config | metric | tree | cap | pre | mi | cap / tree | pre / tree | mi / tree |
|---|---|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1137 (1130–1161) | 1144 (1140–1149) | 1144 (1140–1163) | 1011 (984–1083) | 1.01× | 1.01× | 0.89× |
| loop | 1024B | Mops ↑ | 755 (729–804) | 750 (708–777) | 800 (783–857) | 792 (761–796) | 0.99× | 1.06× | 1.05× |
| loop | 16384B | Mops ↑ | 769 (764–785) | 773 (762–807) | 861 (830–864) | 368 (360–373) | 1.01× | 1.12× | 0.48× |
| xthread | w4,1KiB | Mfree ↑ | 83.5 (82.1–86.6) | 84.5 (83.0–86.0) | 86.6 (82.6–89.4) | 108 (106–113) | 1.01× | 1.04× | 1.29× |
| xmalloc | w4,64B | Mfree ↑ | 396 (394–400) | 404 (400–409) | 394 (385–400) | 345 (342–349) | 1.02× | 1.00× | 0.87× |
| larson | t4 | Mops ↑ | 169 (165–170) | 168 (165–171) | 168 (168–172) | 195 (195–196) | 0.99× | 1.00× | 1.16× |
| rptest | t4 | Mops_cpu ↑ | 7.05 (6.65–7.74) | 7.62 (6.75–7.86) | 6.88 (6.58–7.63) | 6.69 (6.48–7.07) | 1.08× | 0.98× | 0.95× |
| frame | t4 | fps ↑ | 5227 (5113–5244) | 5234 (5099–5260) | 5266 (5146–5284) | 5649 (5627–5663) | 1.00× | 1.01× | 1.08× |
| frame | t4 | frame_max_us ↓ | 285 (220–410) | 399 (271–675) | 328 (242–416) | 397 (304–1347) | 1.40× | 1.15× | 1.39× |
| frame | t4 | rss_mib ↓ | 917 (916–932) | 916 (916–931) | 916 (916–930) | 465 (462–490) | 1.00× | 1.00× | 0.51× |
| xthread | w8,1KiB | Mfree ↑ | 123 (119–130) | 122 (118–124) | 124 (119–126) | 80.6 (77.4–83.2) | 0.99× | 1.00× | 0.65× |
| xmalloc | w8,64B | Mfree ↑ | 510 (500–511) | 511 (495–512) | 508 (506–511) | 504 (489–505) | 1.00× | 1.00× | 0.99× |
| larson | t8 | Mops ↑ | 256 (253–264) | 261 (259–267) | 263 (260–263) | 207 (206–210) | 1.02× | 1.03× | 0.81× |
| rptest | t8 | Mops_cpu ↑ | 6.25 (5.9–6.38) | 6.24 (5.75–6.62) | 6.14 (6.07–6.87) | 5.7 (5.17–5.99) | 1.00× | 0.98× | 0.91× |
| frame | t8 | fps ↑ | 4156 (4134–4175) | 4180 (4157–4198) | 4173 (4097–4204) | 3671 (3568–3697) | 1.01× | 1.00× | 0.88× |
| frame | t8 | frame_max_us ↓ | 446 (438–500) | 458 (415–494) | 450 (400–533) | 1738 (1665–1870) | 1.03× | 1.01× | 3.89× |
| frame | t8 | rss_mib ↓ | 1162 (1161–1162) | 1158 (1157–1158) | 1162 (1158–1163) | 585 (578–598) | 1.00× | 1.00× | 0.50× |
| xlat | 1024Bx200000 | rate_M ↑ | 4.99 (4.89–5.15) | 4.96 (4.94–5.07) | 4.95 (4.92–5.56) | 9.19 (8.97–9.28) | 0.99× | 0.99× | 1.84× |
| xlat | 1024Bx1000000 | max_ns ↓ | 14084 (12792–15750) | 14000 (13708–20000) | 16542 (14709–47500) | 24875 (22750–33292) | 0.99× | 1.17× | 1.77× |

The rest (frameRT, the other frame percentiles, slow calls, the other xlat
rows) is level within its spread; `summary.md` there has every row.  Tree
and cap are level everywhere (the frame t4 max and the rptest t4 spread
overlap).  Against pre, the §tl-cap cost on the single-thread loop stays:
1 KiB 0.94x and 16 KiB 0.89x here, 0.89x and 0.92x in the run above (pre's
own 1 KiB moved 842 → 800 between the runs).

**Micro benches** (`ab.py . 5 pre=… cap=… tree=…`, median of 5):

| config | unit | pre | cap | tree |
|---|---|---|---|---|
| replace1 5000 20000000 400 1000 | M ops/s | 401.7 | 391.6 | 390.1 |
| replace1 5000 20000000 8 1000 | M ops/s | 201.6 | 199.2 | 198.6 |
| replace1 100000 20000000 16 4096 | M ops/s | 127.4 | 122.2 | 122.2 |
| replace1 5000 20000000 1500 3000 | M ops/s | 387.0 | 358.6 | 360.3 |
| replace1 5000 20000000 16 368 | M ops/s | 280.8 | 277.6 | 277.4 |
| heavy 100000 1000000 4 400 1000 | ns/alloc | 73.2 | 75.6 | 75.9 |
| heavy 100000 1000000 32 400 1000 | ns/alloc | 64.5 | 65.1 | 64.9 |
| heavy 100000 1000000 32 32 512 | ns/alloc | 31.6 | 31.3 | 31.1 |
| ring 64 4 20 | M ops/s | 13.3 | 13.2 | 13.1 |
| ring 64 32 20 | M ops/s | 21.9 | 22.7 | 22.3 |
| ring 64 128 20 | M ops/s | 24.7 | 23.9 | 23.1 |

**Hitches.**  The probe of the run above (200 MiB of one size, every free
timed, in allocation order), 3 runs per arm:

| size | metric | pre | cap | tree | mi |
|---|---|---|---|---|---|
| 64 B | mean | 7.3–8.3 ns | 9.6–9.7 ns | 9.6–9.7 ns | 7.3–7.4 ns |
| 64 B | p99.99 | 84–125 ns | 10.29–10.33 µs | 10.08–10.25 µs | 125 ns |
| 64 B | max | 14.0–21.8 µs | 33.2–35.6 µs | 25.2–33.6 µs | 11.8–17.8 µs |
| 64 B | frees > 1 µs / > 10 µs | 14–26 / 7–14 | 805–811 / 423–448 | 804–814 / 373–420 | 18–33 / 8–15 |
| 1 KiB | mean | 27.6–32.0 ns | 55.3–56.6 ns | 54.5–56.3 ns | 8.6–9.6 ns |
| 1 KiB | p99.99 | 209–250 ns | 9.5–10.0 µs | 8.7–9.5 µs | 167–250 ns |
| 1 KiB | max | 14.6–21.3 µs | 19.5–24.4 µs | 18.4–32.1 µs | 0.96–10.7 µs |
| 1 KiB | frees > 1 µs / > 10 µs | 2–7 / 2–7 | 838–845 / 12–21 | 837–845 / 9–17 | 0–1 / 0–1 |

Tree and cap alike: one slow free per returned segment.  This time the
64 B worst free sat ~10 µs above pre's (it overlapped pre's in the run
above).

**Memory (item 6).**  The phase probe, now also freeing the first phase in
a fixed random permutation; peak `phys_footprint`, MiB:

| A → B, order of the frees | pre | cap | tree | mi |
|---|---|---|---|---|
| 64 B → 1 KiB, allocation order | 1053 | 631 | 627 | 481 |
| the same, `kame_pool_trim()` between | 1003 | 581 | 577 | — |
| 64 B → 1 KiB, random order | 1054 | 1054 | **627** | 481 |
| the same, `kame_pool_trim()` between | 1003 | 1003 | **577** | — |
| 4 KiB → 64 B, allocation order | 891 | 707 | 703 | 467 |
| the same, `kame_pool_trim()` between | 890 | 588 | 588 | — |
| 4 KiB → 64 B, random order | 891 | 891 | **703** | 467 |
| the same, `kame_pool_trim()` between | 890 | 829 | **588** | — |

Freed in a random order the cap alone held nothing back (cap = pre);
`tl_reclaim_idle` makes the random order peak where allocation order does,
627 / 577 MiB as on the M3.  In no arm does the footprint drop at the frees
themselves; the trim takes 64 B's 481 MiB to 431 and leaves 4 KiB's 414.

## Since `4ae71a910`: the freed slot goes back, idle reclaim bounded — what to re-check

The sealed segments of `f9e5cb9b3` returned the oldest 256 KiB of a list in
the one free that went past it.  Measured on the M3 with
`tests/bench/micro/freelat.c` (200 MiB of one size freed by one thread),
that free is a walk of slots freed long ago, a dependent miss each: in a
random order 4.9 ms at 16 B and 1.2 ms at 64 B (786 frees over 100 µs), and
130 ns a free on average against 7 ns with no cap.  The M5 runs above timed
allocation order only, where the walk prefetches (8–10 µs a segment).  The
idle reclaim returned whole lists in one malloc the same way, up to 4 MiB a
bucket.

Now, on every thread, a free that would take its list past
`KAME_TL_CAP_BYTES` (4 MiB) returns its own slot to its chunk — the realtime
path of before — touching only that chunk's bitmap word and header.  A slot
below the full tier counts `KAME_TL_SLOT_COST` (1 KiB) more than its size,
so a list of small slots stays short (4 MiB of 16 B is 262144 slots; with the
cost about 4000): it bounds the chunks a list can hold and what the reclaim
must walk.  Before a new chunk, the buckets whose list heads have not moved
since the last one give back ≤ `KAME_TL_RECLAIM_SLOTS` (256) slots between
them, off the heads, on realtime threads too; one interval, no strike bit.
`KameTlsPage` is 432 B again (`m_seg` gone); `KAME_TL_SEG_BYTES` /
`KAME_TL_SEGS` are an `#error`.  The harsh trees now take
`-DKAME_TL_CAP_BYTES=2048u -DKAME_TL_RECLAIM_SLOTS=1`.

M3 (MacBook Air), against master `4ae71a910`:

| | master | this |
|---|---|---|
| freelat random 16 B: mean / max | 135 ns / 4.9 ms | 13 ns / 11 µs |
| freelat random 64 B: mean / max | 129 ns / 1.2 ms | 14 ns / 4.5 µs |
| freelat random 1 KiB: mean / max | 126 ns / 81 µs | 49 ns / 7.5 µs |
| phase change 400 MiB, random order, footprint after phase 2: 16 B → 1 KiB / 64 B → 1 KiB | 777 / 627 MiB | 777 / 627 MiB (no reclaim: 1163 / 1009) |
| bench_loop 64 B / 1 KiB / 16 KiB (9 reps) | 1 | 1.000 / 0.997 / 1.000× |
| bench_ab t4, t8 (5 reps): xthread, xmalloc, larson, rptest, frame fps, xlat | 1 | 0.91–1.03×, every range overlapping; frame p99 / max 0.91 / 0.92× (t4) |
| micro: replace1, heavy, ring (5–11 reps) | 1 | 0.98–1.09× |
| replace1 100 K live 16..4096 B: chunks / max RSS | 513 / 248.8 MiB | 513 / 248.8 MiB |

The slot cost is added only below the full tier: added there too it moved
bench_loop 16 KiB to 0.93× by code placement alone (the add is off the
dependence chain; the loop keeps one slot on the list).  ctest 49/49 in
Release, with asserts, and in the harsh tree; `alloc_phase_test`,
`alloc_fit_test`, `alloc_aligned_test`, `alloc_rt_thread_test`, the
thread-exit tests and the exclusivity tests 40 runs each in Release and
harsh, 0 failures.

To re-check, on the M5 and Linux:

1. **The hitches.**  `freelat 16 200`, `freelat 64 200`, `freelat 1024 200`,
   each in order and with `SHUFFLE=1`, this branch against master: no free
   over 100 µs is the claim.  Linux faults pages differently; `RT=1` should
   change nothing now.
2. **Memory.**  The phase change in a random order (the probe of item 5
   above, freed shuffled): 64 B → 1 KiB, 16 B → 1 KiB, 4 KiB → 64 B.  The
   reclaim is gradual — 256 slots per new chunk — so the peak, not the end,
   is the number.
3. **The hot path.**  `bench_loop` 64 B / 1 KiB / 16 KiB interleaved against
   master; the 16 KiB loop is the one that moved by placement on the M3.
4. **Correctness.**  The Linux trees and the harsh ones with the new flags:
   ctest and the soaks of the runs above.

## Run: M5 Ultra, 2026-10-13 — the freed slot goes back (master `9f6370a52`)

Same machine and toolchain as the runs above.  Arms: tree at `9f6370a52`
(sha256 `83f4b793f1402ed8`), prev = `4ae71a910`'s `kamepoolalloc/` (sha256
`fc5fe9777215ee47`, the tree arm of the run above), pre = `727c47322`'s
(sha256 `6d9157ba1b90e29d`, before the cap), mimalloc 3.3.2.

**Correctness (item 4).**  Full-tree ctest 49/49 in Release, in `-O2 -g`
with asserts, and in that build with `-DKAME_TL_CAP_BYTES=2048u
-DKAME_TL_RECLAIM_SLOTS=1`.  In all three, scored by exit status, 6 in
parallel: `alloc_tsd_exclusivity_test` and `_var_test` 2000 each,
`alloc_fit_test` 300, `alloc_phase_test` 300, `alloc_rt_thread_test` 200,
`alloc_aligned_test` 200, the four `alloc_thread_exit_*` tests 200 each —
0 failures.

**The hitches (item 1).**  `freelat SIZE 200` (`tests/bench/micro/freelat.c`,
`cc -O2`), in allocation order, with `SHUFFLE=1`, with `RT=1`, and with
both; 3 runs per arm (mimalloc has no RT mode):

| size | order | arm | mean ns | p99.99 µs | max µs | > 1 / 10 / 100 µs |
|---|---|---|---|---|---|---|
| 16 B | in order | pre | 4.4–4.5 | 0.08 | 39.8–40.3 | 183–249 / 27–46 / 0 |
| 16 B | in order | prev | 6.0–6.3 | 0.08–0.12 | 57.2–79.8 | 959–1002 / 806–840 / 0 |
| 16 B | in order | tree | 8.8–9.2 | 0.17–0.21 | 39.9–47.7 | 825–877 / 50–91 / 0 |
| 16 B | in order | mi | 4.2–4.5 | 0.08 | 16.9–40.2 | 87–198 / 18–37 / 0 |
| 16 B | random | pre | 4.8–5.0 | 0.12 | 18.6–21.0 | 49–182 / 7–29 / 0 |
| 16 B | random | prev | 120.1–120.4 | 0.17 | 2059.3–2089.5 | 896–919 / 792–812 / 784 |
| 16 B | random | tree | 10.1–11.1 | 0.17 | 18.5–27.8 | 227–312 / 14–60 / 0 |
| 16 B | random | mi | 4.2–4.3 | 0.12 | 18.5–72.0 | 69–125 / 8–38 / 0 |
| 16 B | in order RT | pre | 4.2–4.5 | 0.08 | 25.8–37.3 | 101–288 / 8–11 / 0 |
| 16 B | in order RT | prev | 9.0–9.7 | 0.17–0.33 | 15.6–38.2 | 194–513 / 10–17 / 0 |
| 16 B | in order RT | tree | 8.5–9.2 | 0.17–0.33 | 14.7–39.5 | 201–867 / 12–14 / 0 |
| 16 B | random RT | pre | 4.9–5.0 | 0.08–0.12 | 19.8–24.0 | 73–131 / 7–30 / 0 |
| 16 B | random RT | prev | 10.3–11.2 | 0.17 | 16.2–22.0 | 111–311 / 12–70 / 0 |
| 16 B | random RT | tree | 10.2–10.8 | 0.17–0.25 | 18.3–26.5 | 122–245 / 11–57 / 0 |
| 64 B | in order | pre | 4.2 | 0.08–0.12 | 9.0–22.9 | 23–35 / 0–3 / 0 |
| 64 B | in order | prev | 6.7 | 10.71–10.96 | 20.2–29.3 | 802–831 / 496–517 / 0 |
| 64 B | in order | tree | 9.2 | 3.21–3.38 | 14.1–17.7 | 622–625 / 5–8 / 0 |
| 64 B | in order | mi | 4.2–4.3 | 0.08–0.12 | 7.2–15.1 | 16–42 / 0–2 / 0 |
| 64 B | random | pre | 5.0–5.2 | 0.08–0.12 | 7.8–20.2 | 15–40 / 0–4 / 0 |
| 64 B | random | prev | 100.5–104.1 | 423.46–436.00 | 491.1–658.0 | 813–828 / 784–788 / 784 |
| 64 B | random | tree | 10.2–11.0 | 0.17 | 14.1–23.8 | 48–112 / 2–5 / 0 |
| 64 B | random | mi | 4.3 | 0.17 | 14.7–16.8 | 13–39 / 1–3 / 0 |
| 64 B | in order RT | pre | 4.2 | 0.08–0.12 | 8.0–18.5 | 18–40 / 0–6 / 0 |
| 64 B | in order RT | prev | 9.1–9.2 | 0.71 | 10.2–15.9 | 34–96 / 1–10 / 0 |
| 64 B | in order RT | tree | 8.7 | 0.71 | 15.5–16.5 | 52–83 / 4–15 / 0 |
| 64 B | random RT | pre | 4.9–5.2 | 0.08–0.12 | 15.8–23.0 | 33–44 / 4–10 / 0 |
| 64 B | random RT | prev | 10.0–11.3 | 0.17–0.21 | 17.7–20.6 | 92–172 / 12–22 / 0 |
| 64 B | random RT | tree | 10.6–11.3 | 0.17–0.25 | 16.3–30.0 | 41–154 / 11–22 / 0 |
| 1024 B | in order | pre | 11.1–12.2 | 0.21–0.29 | 10.1–15.8 | 2–12 / 1–3 / 0 |
| 1024 B | in order | prev | 35.2–37.3 | 10.00–13.04 | 15.7–26.4 | 833–842 / 20–46 / 0 |
| 1024 B | in order | tree | 37.8–40.5 | 3.88–9.12 | 15.9–44.6 | 847–867 / 6–15 / 0 |
| 1024 B | in order | mi | 6.1–6.6 | 0.21–0.29 | 9.1–11.0 | 4–12 / 0–2 / 0 |
| 1024 B | random | pre | 17.9–18.4 | 0.21–0.29 | 0.4–15.2 | 0–15 / 0–3 / 0 |
| 1024 B | random | prev | 69.9–72.4 | 19.62–22.12 | 30.7–36.1 | 835–837 / 570–737 / 0 |
| 1024 B | random | tree | 34.0–34.6 | 0.33–0.42 | 11.2–11.7 | 5–15 / 2–4 / 0 |
| 1024 B | random | mi | 6.7–7.0 | 0.25–0.29 | 1.1–5.6 | 1–2 / 0 / 0 |
| 1024 B | in order RT | pre | 10.9–11.5 | 0.21 | 1.9–24.7 | 1–3 / 0–1 / 0 |
| 1024 B | in order RT | prev | 30.6–34.5 | 0.71–7.29 | 10.3–16.9 | 12–816 / 3–7 / 0 |
| 1024 B | in order RT | tree | 28.8–29.0 | 0.71–2.21 | 11.0–17.8 | 7–21 / 1–5 / 0 |
| 1024 B | random RT | pre | 17.7–18.1 | 0.21–0.25 | 11.3–13.7 | 1–13 / 1–3 / 0 |
| 1024 B | random RT | prev | 39.3–39.4 | 0.62–0.79 | 15.2–19.8 | 6–16 / 3–8 / 0 |
| 1024 B | random RT | tree | 34.1–34.7 | 0.29–0.42 | 0.8–12.5 | 0–11 / 0–3 / 0 |

The claim holds on the M5: tree has no free over 100 µs in any size, order
or mode (worst 48 µs, 16 B in allocation order; pre's worst there is 40 µs).
prev, freed in a random order, has 784 frees over 100 µs at 16 B and at
64 B, up to 2.09 ms and 0.66 ms (the M3's 4.9 ms and 1.2 ms), and a mean of
120 / 100 ns; tree's random-order mean is 10–11 ns at 16 and 64 B and 34 ns
at 1 KiB.  The mean is still about twice pre's (in order 9 vs 4.4 ns at
16 B, 38–40 vs 11–12 ns at 1 KiB): past the cap every one of these frees
returns its slot.  `RT=1` changes nothing in a random order.  In
allocation order it does at 64 B and 1 KiB: without it 622–625 and 847–867
frees take over 1 µs (p99.99 3.2–3.4 and 3.9–9.1 µs), with it 52–83 and
7–21 (p99.99 0.71 and 0.71–2.2 µs).  At 16 B both have hundreds (825–877
without, 201–867 with; p99.99 ≤ 0.33 µs either way).  The counts are near
200 MiB / 256 KiB = 800, one per chunk unit, so these are likely the frees
that empty a chunk and release it, which a realtime thread leaves for
later; not traced.

**Memory (item 2).**  The phase probe of the runs above (one thread, 400 MiB
of A-byte blocks, freed in allocation order or a fixed random permutation,
`kame_pool_trim()` between where marked, then 400 MiB of B-byte blocks);
peak `phys_footprint`, MiB:

| A → B, order of the frees | pre | prev | tree | mi |
|---|---|---|---|---|
| 64 B → 1 KiB, allocation order | 1053 | 627 | 627 | 481 |
| the same, trim between | 1003 | 577 | 577 | — |
| 64 B → 1 KiB, random order | 1053 | 627 | 627 | 481 |
| the same, trim between | 1003 | 577 | 577 | — |
| 16 B → 1 KiB, allocation order | 1204 | 777 | 777 | 641 |
| the same, trim between | 1003 | 663 | 663 | — |
| 16 B → 1 KiB, random order | 1204 | 777 | 777 | 641 |
| the same, trim between | 1003 | 663 | 663 | — |
| 4 KiB → 64 B, allocation order | 891 | 703 | 703 | 467 |
| the same, trim between | 890 | 588 | 588 | — |
| 4 KiB → 64 B, random order | 891 | 703 | 703 | 467 |
| the same, trim between | 890 | 588 | 588 | — |

tree = prev in every row, and random order = allocation order; 777 / 627 MiB
as on the M3.  With the trim, 16 B → 1 KiB ends at 577 MiB but peaks at 663
before it (prev the same).

**The hot path (item 3).**  `bench_ab.sh --only loop --reps 9` (results
`~/bench-ab/results-dhcp-029-018-20261013-131218`):

| bench | config | metric | tree | prev | pre | prev / tree | pre / tree |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1124 (1092–1136) | 1143 (1123–1152) | 1147 (1125–1204) | 1.02× | 1.02× |
| loop | 1024B | Mops ↑ | 748 (705–783) | 731 (711–796) | 826 (785–855) | 0.98× | 1.11× |
| loop | 16384B | Mops ↑ | 773 (756–856) | 775 (764–805) | 834 (812–901) | 1.00× | 1.08× |

Level with prev (16 KiB included); against pre the cap's 1 KiB / 16 KiB
cost of the runs above stays.

**`bench_ab.sh`** (the set of the runs above, `--threads "4 8" --reps 5`;
results `~/bench-ab/results-dhcp-029-018-20261013-124512`):

| bench | config | metric | tree | prev | pre | mi | prev / tree | pre / tree | mi / tree |
|---|---|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1134 (1100–1137) | 1141 (1114–1153) | 1148 (1128–1163) | 1048 (999–1074) | 1.01× | 1.01× | 0.92× |
| loop | 1024B | Mops ↑ | 769 (725–831) | 760 (718–776) | 791 (766–835) | 774 (731–799) | 0.99× | 1.03× | 1.01× |
| loop | 16384B | Mops ↑ | 768 (761–836) | 788 (763–816) | 847 (823–863) | 363 (358–369) | 1.03× | 1.10× | 0.47× |
| xthread | w4,1KiB | Mfree ↑ | 88.4 (85.3–88.9) | 86.5 (84.7–88.3) | 87.4 (85.2–89.7) | 102 (98.8–108) | 0.98× | 0.99× | 1.15× |
| xmalloc | w4,64B | Mfree ↑ | 393 (382–399) | 393 (391–402) | 386 (298–393) | 346 (343–349) | 1.00× | 0.98× | 0.88× |
| larson | t4 | Mops ↑ | 164 (164–165) | 168 (167–172) | 166 (161–167) | 194 (193–195) | 1.03× | 1.01× | 1.18× |
| rptest | t4 | Mops_cpu ↑ | 7.37 (7.26–7.93) | 7.14 (6.77–7.73) | 7.75 (7.37–8.43) | 7.04 (6.81–7.75) | 0.97× | 1.05× | 0.96× |
| frame | t4 | fps ↑ | 4859 (4617–4894) | 4851 (4720–4894) | 4906 (4892–4994) | 5327 (5317–5357) | 1.00× | 1.01× | 1.10× |
| frame | t4 | frame_max_us ↓ | 365 (325–401) | 399 (345–434) | 338 (324–389) | 425 (342–1009) | 1.09× | 0.92× | 1.16× |
| frame | t4 | call_max_ns ↓ | 80219 (61326–93093) | 64950 (50826–77409) | 77576 (48285–91676) | 192062 (114112–825324) | 0.81× | 0.97× | 2.39× |
| frameRT | t4 | call_max_ns ↓ | 80176 (68034–91344) | 55993 (54642–71093) | 73326 (57159–99510) | 216205 (139403–726379) | 0.70× | 0.91× | 2.70× |
| frame | t4 | rss_mib ↓ | 934 (931–943) | 935 (932–943) | 930 (930–931) | 485 (466–497) | 1.00× | 1.00× | 0.52× |
| xthread | w8,1KiB | Mfree ↑ | 128 (123–129) | 125 (123–140) | 128 (123–130) | 84.7 (81.5–88.3) | 0.97× | 0.99× | 0.66× |
| xmalloc | w8,64B | Mfree ↑ | 500 (495–503) | 498 (497–501) | 499 (494–502) | 497 (472–500) | 1.00× | 1.00× | 1.00× |
| larson | t8 | Mops ↑ | 247 (242–252) | 254 (246–256) | 256 (252–259) | 204 (194–206) | 1.03× | 1.03× | 0.82× |
| rptest | t8 | Mops_cpu ↑ | 6.45 (5.92–6.74) | 6.58 (6.48–7.16) | 6.28 (6.06–6.41) | 5.52 (5.31–5.99) | 1.02× | 0.97× | 0.86× |
| frame | t8 | fps ↑ | 3864 (3806–3884) | 3888 (3866–3948) | 3885 (3629–3922) | 3518 (3384–3561) | 1.01× | 1.01× | 0.91× |
| frame | t8 | frame_max_us ↓ | 493 (480–498) | 486 (465–508) | 497 (481–505) | 1171 (1063–1726) | 0.99× | 1.01× | 2.37× |
| frame | t8 | call_max_ns ↓ | 115303 (99009–159337) | 96177 (89302–150087) | 105445 (68577–128194) | 1075989 (936336–1481361) | 0.83× | 0.91× | 9.33× |
| frameRT | t8 | call_max_ns ↓ | 95886 (78468–133778) | 106426 (74492–298382) | 102653 (73052–140087) | 1057406 (1012579–1583455) | 1.11× | 1.07× | 11.03× |
| frame | t8 | rss_mib ↓ | 1166 (1162–1180) | 1166 (1162–1166) | 1163 (1159–1176) | 593 (584–614) | 1.00× | 1.00× | 0.51× |
| xlat | 1024Bx200000 | rate_M ↑ | 4.92 (4.85–5.11) | 4.99 (4.84–5.08) | 4.96 (4.89–5.27) | 9.31 (8.88–9.36) | 1.01× | 1.01× | 1.89× |
| xlat | 1024Bx1000000 | max_ns ↓ | 32125 (14792–37667) | 16750 (15166–37916) | 17209 (14875–48458) | 35833 (26708–45792) | 0.52× | 0.54× | 1.12× |

The rest is level within its spread (`summary.md` there).  Every arm,
mimalloc included, ran frame ~7 % below the run above (machine state); the
comparison is within the run.  Tree against prev overlaps everywhere but
larson t4 (164 vs 168, 0.97x; at t8 the ranges overlap).  frame / frameRT
`call_max_ns` at t4 has the higher median (80 vs 56–65 µs) but overlapping
ranges, and at t8 it goes both ways.

**Micro benches** (`ab.py . 5 pre=… prev=… tree=…`, median of 5):

| config | unit | pre | prev | tree |
|---|---|---|---|---|
| replace1 5000 20000000 400 1000 | M ops/s | 398.4 | 389.3 | 385.5 |
| replace1 5000 20000000 8 1000 | M ops/s | 200.6 | 197.5 | 199.5 |
| replace1 100000 20000000 16 4096 | M ops/s | 118.3 | 113.5 | 114.5 |
| replace1 5000 20000000 1500 3000 | M ops/s | 379.7 | 352.8 | 360.3 |
| replace1 5000 20000000 16 368 | M ops/s | 277.3 | 265.2 | 263.1 |
| heavy 100000 1000000 4 400 1000 | ns/alloc | 80.7 | 82.7 | 82.1 |
| heavy 100000 1000000 32 400 1000 | ns/alloc | 68.3 | 69.7 | 68.1 |
| heavy 100000 1000000 32 32 512 | ns/alloc | 33.1 | 33.2 | 33.3 |
| ring 64 4 20 | M ops/s | 13.4 | 13.2 | 13.0 |
| ring 64 32 20 | M ops/s | 21.9 | 22.4 | 22.3 |
| ring 64 128 20 | M ops/s | 24.0 | 23.9 | 24.1 |

tree within 1–2 % of prev throughout.

## Run: Linux, 2026-10-13 (master `9f6370a52`; the ILP32 re-run of `75369a9a3` / `10e5bd4bf` and the re-check of the freed-slot return)

Same container (x86-64, g++ 13.3, 4 vCPU) and six trees as the Linux runs
above, rebuilt at `9f6370a52`, plus harsh trees with the new flags
`-DKAME_TL_CAP_BYTES=2048u -DKAME_TL_RECLAIM_SLOTS=1` (LP64 release, LP64
asserts, i586 asserts; flags confirmed in their caches).  Library-only builds
at `460730c35` (LP64, i586, i486) for the alignment bug and at `4ae71a910`
(LP64) for the freed-slot comparison.

**ILP32, items 1–4 of "Since the Linux runs".**

1. *Build without `-k`*: all nine trees build whole, `bench_frame` included
   on i486.
2. *ctest*: LP64 49/49 ×2, harsh LP64 49/49 ×2, i586 asserts 49/49, i486
   asserts 42/42; i586 release 48/49, i486 release 41/42 and harsh i586
   48/49 — `transaction_wait_budget_test` under `ctest -j3` each time, 3/3
   PASS alone.  `alloc_fit_test`, `alloc_phase_test` and `alloc_aligned_test`
   pass on every tree.  `alloc_fit_test`'s ALIGN 128 tier on ILP32 (256 B
   fillers, 128 B probes) — §fit's first run there: **299 / 3000 probes
   among the fillers** (rounds 1 / 2), 0 short, 0 overwritten; its ALIGN 32
   tier 1499 / 3000.  No-DCAS 3/3.
3. *The alignment bug at run time*, `alloc_aligned_test` (built at
   `9f6370a52`) against each library by `LD_LIBRARY_PATH`, 3 runs each:

   | | own library (`9f6370a52`) | `460730c35` library |
   |---|---|---|
   | LP64 | PASS ×3 | PASS ×3 |
   | i586 | PASS ×3 | **FAIL ×3**: "misaligned: alignment 256, size 1 -> 0xf410e180" |
   | i486 | PASS ×3 | **FAIL ×3**: the same |

   The failing addresses are 128-aligned, not 256 — the every-other-slot
   misalignment `75369a9a3` read off the tables, now seen.
4. *Soaks*: `alloc_phase_test`, `alloc_fit_test` and `alloc_aligned_test`,
   **0 / 100** each on all four ILP32 trees.

**Item 4 of "Since `4ae71a910`" (correctness).**  Soaks by exit status, 4 at
a time, a stage's tests interleaved, a SIGSEGV reporter on every run:

| tree | phase / rt_thread / aligned / 4 thread-exit | exclusivity 32 B / var / mid | fit |
|---|---|---|---|
| LP64 release | 0 / 200 each | 0 / 500 each | 0 / 500 |
| harsh LP64 release, harsh LP64 asserts, harsh i586 asserts | 0 / 100 each (phase, rt, 4 thread-exit) | 0 / 100 each | 0 / 100 |

**Item 1 (hitches).**  `freelat SIZE 200`, LP64, `LD_PRELOAD` of each
library, interleaved, 3 runs each; frees over 100 µs (and p99.99):

| | `9f6370a52` | `4ae71a910` |
|---|---|---|
| 16 B in order | 21–45 (17–22 µs) | 724–761 (21–23 µs) |
| 16 B `SHUFFLE=1` | 47–67 (26–31 µs) | 830–899 (39–46 µs) |
| 64 B in order | 0–19 (23–25 µs) | 90–201 (82–90 µs) |
| 64 B `SHUFFLE=1` | 3–23 (21–36 µs) | 799–822 (**934–1095 µs**) |
| 1 KiB in order | 0–1 (44–49 µs) | 41–131 (111–143 µs) |
| 1 KiB `SHUFFLE=1` | 0–12 (35–50 µs) | 11–82 (91–137 µs) |

Frees over 100 µs drop 10–200×, and the 64 B random-order p99.99 from ~1 ms
to ~25 µs.  "No free over 100 µs" does not hold literally here: a few dozen
of 13 M frees still exceed it with the new library, and the maxima (to
9.9 ms, in both arms) are what a shared VM's preemption looks like; without
a profiler they are not attributed.  `RT=1` with `SHUFFLE=1` now matches
`SHUFFLE=1` alone in every size (16 B mean 159–168 vs 140–178 ns), as
claimed; the old library's RT path differed (16 B 151–177 vs 350–405 ns).
The in-order mean rose at 16 B (50–60 vs 38–39 ns) and a little at 64 B
(50–58 vs 46–50 ns); at 1 KiB it fell (304–317 vs 427–473 ns).

**Item 2 (memory).**  The phase-change probe of the runs above, freed in a
random order (`SHUFFLE=1`), peak RSS (VmHWM), one thread, 400 MiB a phase:

| phase change | `9f6370a52` | `4ae71a910` | glibc |
|---|---|---|---|
| 64 B → 1 KiB | 600 MiB | 600 | 553 |
| 16 B → 1 KiB | 750 | 750 | 1003 |
| 4 KiB → 64 B | 679 | 679 | 553 |

Identical peaks for the two libraries, as on the M3 (777 / 627 there, same
before and after).  Item 3 (`bench_loop`) is timing on a real host; not
measured on this VM.

## Run: Windows, Xeon E5-1630 v4, 2026-10-13 (master `5d2f5a726`)

Intel Xeon E5-1630 v4 (Broadwell-EP, 4 cores / 8 threads, 3.7 GHz), Windows
11 Pro 26200, Balanced power plan, a desktop in use (browser open, CPU
8–15 % before each run).  llvm-mingw clang 17, `-O3 -DNDEBUG`, the
`kamepoolalloc/tests` CMake tree; `bench_loop_pool` (`kame_pool_malloc`
directly — PE/COFF has no preload) against a `libkamepoolalloc.dll` beside
it, so each arm is its own build directory of one commit.  This is item 3
(`bench_loop`) on a real x86 host, which the Linux runs above left open.

**Correctness.**  clang x64: pool 25/25, STM 21/22.  llvm-mingw i686: pool
24/25, STM 21/22.  cl x64: pool 17/25, STM 14/22; cl x86: pool 16/25, STM
14/22 (cl does not compile 8 pool and 7 STM test targets, the same set
since Aug 28; each STM count also has the wait-budget failure below, and
x86's pool count `alloc_huge_test`).  `check_no_dcas.sh` 3/3; kame.exe and every module
build.  The two failures are host conditions, not the allocator:
`alloc_huge_test` on 32-bit (its MT step does not fit 2 GB; passes once
`/LARGEADDRESSAWARE`), and `transaction_wait_budget_test`, whose 10 ms arm
landed on the 15.6 ms timer tick (p99.99 15.6–15.8 ms in three serial runs)
— `condition_variable::wait_for(10 ms)` took 15.6 ms on this host at the
time, 10.2 ms under `timeBeginPeriod(1)`.

**`bench_compare.sh`** at `5d2f5a726`, median of 5 — now the README's
Windows tables.  Against them, the README's previous numbers (`8f2e6980`,
June):

| size | 1T June | 1T now | 4 proc June | 4 proc now |
|---|---|---|---|---|
| 64 B | 301 | 226 | 1055 | 852 |
| 1 KiB | 223 | 228 | | |
| 16 KiB | 191 | 124 | 687 | 473 |
| 1 MiB | 64 | 63 | 231 | 208 |

**Interleaved A/B**, same machine state, every arm at every size each rep,
arm order alternating, median (min–max).  `8f2e6980` = the June table,
`45f1d07f9` = the parent of `5fc34e052`.  Single thread, 7 reps:

| size | `8f2e6980` | `45f1d07f9` | `5fc34e052` | `5d2f5a726` |
|---|---|---|---|---|
| 64 B | 275 (223–305) | 297 (211–311) | 232 (217–253) | 229 (205–241) |
| 1 KiB | 220 (204–245) | 207 (193–221) | 272 (249–292) | 244 (236–258) |
| 16 KiB | 177 (172–193) | 183 (177–195) | 128 (94–148) | 117 (108–128) |

4 processes aggregate, 5 reps:

| size | `8f2e6980` | `45f1d07f9` | `5fc34e052` | `5d2f5a726` |
|---|---|---|---|---|
| 64 B | 951 (921–1069) | 952 (906–1098) | 927 (856–958) | 868 (768–877) |
| 16 KiB | 703 (613–726) | 670 (589–694) | 546 (509–597) | 463 (409–478) |
| 1 MiB | 212 (150–219) | 211 (191–227) | 215 (198–220) | 210 (201–225) |

`git bisect` (median of 5 per commit, threshold midway) names `5fc34e052`
for the single-thread 16 KiB step (`45f1d07f9` 191, `5fc34e052` 119) and
`9f6370a52` for the further 4-process 16 KiB step (`f53394135` 524,
`9f6370a52` 461).  Interleaved, `9f6370a52` against `f53394135`: single
thread 64 B 1.08×, 1 KiB 1.06×, 16 KiB 0.97×; 4 processes 64 B 1.04×,
16 KiB 0.92×, 1 MiB 0.99×.

So on this host `5fc34e052` is 64 B 0.78×, 1 KiB 1.31×, 16 KiB 0.70× single
thread, where the M5 Ultra run above measured 0.97× / 0.94× / 1.00× against
v1.2.1.  Not yet known: whether it is x86 or Broadwell (Ohtaka's Zen 2 item
would say), or code placement, which moved the M3's 16 KiB loop by 7 % on
its own.  1 MiB is unchanged throughout; the June table's 231 was the
machine of that day.

## Run: Linux, Intel Xeon @ 2.80 GHz, 2026-10-13 — the Windows regressions checked on another x86

The Windows run's open question: x86, Broadwell, or code placement?  Cloud
VM, 4 vCPU of an Intel Xeon @ 2.80 GHz (1 MiB L2 / core, 33 MiB L3 — a newer
server core than Broadwell-EP), Linux, shared host, so absolute numbers are
noisy and only the interleaved ratios are read.  One `bench_loop` binary
(plain `malloc` / `free`, `gcc -O2`), the arm switched by `LD_PRELOAD` of each
commit's `libkamepoolalloc.so` (`-O3 -DNDEBUG`, built with gcc 13.3 and with
clang 18.1), a glibc arm as the control (the pool arms run ~2× it, so the
preload takes); every arm at every size each rep, arm order reversed every
other rep; 7 reps single thread, 5 reps 4 processes aggregate.  Arms as in the
Windows run: `8f2e6980` (June), `45f1d07f9` (parent of `5fc34e052`),
`5fc34e052`, `f53394135` → `9f6370a52`, and `e2c9e6dde` (HEAD).

Median M ops/s (min–max):


**gcc, single thread** (M ops/s)
| size | glibc | 8f2e6980 (June) | 45f1d07f9 | 5fc34e052 | f53394135 | 9f6370a52 | e2c9e6dde (HEAD) | 5fc/45f | 9f6/f53 | HEAD/June |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 B | 129 (116-141) | 267 (259-285) | 273 (225-279) | 211 (199-224) | 202 (177-228) | 208 (149-223) | 222 (151-232) | 0.77x | 1.03x | 0.83x |
| 1 KiB | 142 (95-147) | 215 (113-227) | 191 (167-215) | 200 (161-218) | 171 (146-194) | 179 (115-208) | 202 (190-211) | 1.05x | 1.04x | 0.94x |
| 16 KiB | 49 (46-54) | 161 (132-172) | 156 (128-170) | 168 (160-175) | 139 (112-169) | 152 (132-171) | 162 (117-174) | 1.08x | 1.09x | 1.01x |

**gcc, 4 processes aggregate** (M ops/s)
| size | glibc | 8f2e6980 (June) | 45f1d07f9 | 5fc34e052 | f53394135 | 9f6370a52 | e2c9e6dde (HEAD) | 5fc/45f | 9f6/f53 | HEAD/June |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 B | 495 (342-526) | 1075 (762-1088) | 995 (962-1041) | 808 (770-836) | 748 (691-826) | 755 (740-811) | 739 (636-877) | 0.81x | 1.01x | 0.69x |
| 16 KiB | 184 (168-210) | 619 (483-636) | 645 (585-652) | 645 (630-658) | 656 (514-699) | 623 (614-658) | 614 (585-692) | 1.00x | 0.95x | 0.99x |
| 1 MiB | 210 (184-221) | 286 (244-309) | 202 (177-218) | 213 (154-229) | 215 (179-237) | 210 (179-218) | 214 (195-221) | 1.06x | 0.98x | 0.75x |

**clang, single thread** (M ops/s)
| size | glibc | 8f2e6980 (June) | 45f1d07f9 | 5fc34e052 | f53394135 | 9f6370a52 | e2c9e6dde (HEAD) | 5fc/45f | 9f6/f53 | HEAD/June |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 B | 139 (119-148) | 314 (284-326) | 281 (236-327) | 239 (166-253) | 226 (201-242) | 228 (211-242) | 224 (202-239) | 0.85x | 1.01x | 0.71x |
| 1 KiB | 136 (111-146) | 208 (140-217) | 201 (160-223) | 240 (175-254) | 211 (178-230) | 204 (169-252) | 238 (204-251) | 1.19x | 0.96x | 1.14x |
| 16 KiB | 49 (46-54) | 191 (165-194) | 191 (161-199) | 190 (165-195) | 181 (177-194) | 181 (170-191) | 180 (156-199) | 0.99x | 1.00x | 0.94x |

**clang, 4 processes aggregate** (M ops/s)
| size | glibc | 8f2e6980 (June) | 45f1d07f9 | 5fc34e052 | f53394135 | 9f6370a52 | e2c9e6dde (HEAD) | 5fc/45f | 9f6/f53 | HEAD/June |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 B | 526 (472-563) | 1229 (1106-1252) | 1086 (916-1178) | 913 (751-962) | 882 (816-975) | 891 (768-923) | 884 (768-915) | 0.84x | 1.01x | 0.72x |
| 16 KiB | 184 (165-199) | 697 (658-760) | 727 (690-756) | 682 (649-747) | 678 (602-695) | 694 (674-740) | 656 (539-730) | 0.94x | 1.02x | 0.94x |
| 1 MiB | 184 (171-225) | 230 (209-246) | 178 (171-237) | 186 (158-191) | 191 (182-218) | 173 (146-228) | 216 (155-226) | 1.04x | 0.91x | 0.94x |

| step | Windows (Broadwell, clang 17) | here, gcc 13 | here, clang 18 |
|---|---|---|---|
| `5fc34e052` / `45f1d07f9`, 1T 64 B | 0.78× | **0.77×** | **0.85×** |
| — 1T 1 KiB | 1.31× | 1.05× | 1.19× |
| — 1T 16 KiB | **0.70×** | 1.08× | 0.99× |
| — 4P 64 B | 0.97× | 0.81× | 0.84× |
| — 4P 16 KiB | 0.81× | 1.00× | 0.94× |
| `9f6370a52` / `f53394135`, 4P 16 KiB | 0.92× | 0.95× | 1.02× |

**64 B reproduces; 16 KiB does not.**  The thread lists' 64 B single-thread
loss is here with both compilers (0.77× / 0.85×, the gcc figure the Windows
one to the percent), so it is not Broadwell's or Windows'.  The 16 KiB loss
(0.70×, and 0.81× on 4 processes) does not appear on this core with either
compiler (0.99–1.08×); with the M5's 1.00× that leaves Broadwell, Windows
(llvm-mingw's TLS and DLL calls) or that build's code placement — not the
algorithm on x86 as such.  `9f6370a52`'s 4-process 16 KiB step is within
noise here (0.95× / 1.02×).

**One step Windows did not show: 1 MiB, since June.**  gcc, 9 reps
interleaved: single thread `8f2e6980` 77 (55–82), `45f1d07f9` 57 (52–60),
HEAD 61 (32–64); 4 processes 277 (223–300), 213 (166–224), 228 (206–247).
Windows measured 212 / 211 / 210 there.  `git bisect run` over the 48
allocator commits between them (median of 5 single-thread runs, threshold 67)
names **`6fe90a931`** (§75: realtime per-thread gating, prewarm, drain,
pre-reserve); interleaved against its parent `1f6a59fd3`, 9 reps: single
thread 74 → 63 (0.86×), 4 processes 259 → 220 (0.85×).  That is the first and
largest identified part of the 0.74×; the rest lies later in the range and is
not bisected.  clang's 4-process 1 MiB row is too noisy here to read.
