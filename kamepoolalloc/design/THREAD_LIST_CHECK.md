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
