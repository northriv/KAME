# Handoff: v1.2.0 against v1.1.0 — where it is faster, and where it is slower

Started 2026-10-10 on a MacBook Air M3.  The regressions reproduced on the
M5 Ultra (§4.1); their causes and the fix are in §5 (2026-10-11), which
measures at or above v1.1.0 on every workload here.  **Open:** Ohtaka (§4),
and the pin decision at the end of §4.  Append results here.

## 1. Summary

v1.2.0 removes the chunk-list walks from the allocation path, and on the
workload that walk hurt it is two orders of magnitude faster.  It is also
slower than v1.1.0 in two places nobody had measured:

- **larson** (mimalloc-bench; thread churn with cross-thread frees):
  0.67–0.74x.  In two steps, §fl-avail and §revive.
- **bench_loop at 1 KiB and 16 KiB** (single thread): 0.85–0.93x; 64 B is
  unchanged.  §fl-avail, partly recovered by §revive, and then the TSan
  relaxed atomics, which cost 0.94x on their own.

Two published statements rest on a narrower measurement than they read:

- The v1.2.0 tag message: "single-thread hot paths within +-5 %".  That is
  §revive's own number — `3b353c5e6` → `39edad5a9`, M5 Ultra,
  `bench_loop_pool` — not v1.1.0 → v1.2.0.
- The drafted mimalloc-bench pin PR: "single-thread paths are unchanged".
  If it is not posted yet, hold it until §4 is in.

## 2. How to reproduce

`tests/bench/bench_ab.sh` (`32ff4c601`) builds both tags with
mimalloc-bench's recipe, runs every workload as one binary with the arm
switched by preload, and alternates the arm order between repetitions.

    tests/bench/bench_ab.sh --threads "4 8"                     # all workloads
    tests/bench/bench_ab.sh --only "loop larson" --threads "2 4"   # the two regressions

Bisecting: build the library at intermediate commits and give them as arms
(`--tags "" --arm NAME=LIB ...`); see `tests/bench/README.md`.  The split
commits are on `standalone/kamepoolalloc`; the table in §3.2 names the KAME
commits they mirror.

## 3. MacBook Air M3

Apple M3 (4P + 4E), 24 GiB, macOS 26.7, AC power; Apple clang 17.0.0; both
libraries Release, `DYLD_INSERT_LIBRARIES`.  M ops/s, higher is better;
median (min–max).  A fanless laptop: treat single-digit percentages with
care — §3.2 carries its own noise estimate.

### 3.1 v1.1.0 against v1.2.0 (5 repetitions)

| workload | v1.1.0 | v1.2.0 | ratio |
|---|---|---|---|
| bench_loop 64 B | 687 (508–703) | 687 (649–742) | 1.00x |
| bench_loop 1 KiB | 489 (474–497) | 435 (422–440) | **0.89x** |
| bench_loop 16 KiB | 558 (515–577) | 497 (476–498) | **0.89x** |
| larson, 2 threads | 58.1 (54.5–59.7) | 38.9 (35.0–41.2) | **0.67x** |
| larson, 4 threads | 98.5 (84.1–103) | 73.3 (63.8–76.1) | **0.74x** |

`bench_xlatency`, 1 KiB blocks, 200 K live, one run each: v1.1.0 median
44 µs, p99.9 108 µs, max 322 µs, 0.022 M allocs/s; v1.2.0 median 42 ns (one
clock step), p99.9 0.79 µs, max 49 µs, 6.3 M allocs/s.

Smoke only (one repetition of 1 s; not a result): bench_xthread 2 workers
1.99x, xmalloc-test 1.04x, rptest 1.01x.

### 3.2 Bisection (3 repetitions)

| arm | KAME commit | larson, 2 threads | bench_loop 1 KiB | bench_loop 16 KiB |
|---|---|---|---|---|
| v1.1.0 | — | 55.0 (54.9–59.0) | 472 | 563 |
| §fl-avail | `3b353c5e6` | 45.1 (39.4–46.9) — **0.82x** | 384 — 0.81x | 479 — 0.85x |
| §revive | `39edad5a9` | 38.8 (38.0–40.2) — **0.71x** | 444 | 516 |
| orphan-chain serial | `1dbf9e63c` | 38.3 | 448 | 510 |
| §group | `bd7875d06` | 39.0 | 459 | 520 |
| owner-id offsets | `ee9d757b0` | 38.2 | 457 | 520 |
| TSan atomics | `d6cc5007d` | 40.2 | 435 | 481 |
| v1.2.0 | — | 39.8 | 437 | 476 |

The last two arms are **the same binary** — the commits between them are
documentation and a load that compiles identically on arm64, and the two
dylibs' sha256 match — so their spread (1–3 %) is this machine's noise.

The TSan step alone, `ee9d757b0` against `d6cc5007d`, 9 repetitions:
bench_loop 1 KiB 464 (448–466) → 436 (431–438), 16 KiB 528 (487–531) →
495 (477–496), both 0.94x; 64 B 1.00x.  `23d843e23` turned the bitmap-word
loads, the `m_sizes` store and the slot-header store/load into
`__atomic_*_n(RELAXED)`: the same instructions on arm64, but the compiler
may no longer merge, reorder or keep them in registers.  That is the
likely cost — not yet checked in the disassembly.

## 4. Still to run

- **M5 Ultra** (where §revive's numbers came from):
  `tests/bench/bench_ab.sh --threads "4 8"`; paste `summary.md` and
  `provenance.txt` here.  Run 2026-10-10: §4.1.
- **Ohtaka** (the README's x86-64 reference):

      CC=~/llvm-install/bin/clang CXX=~/llvm-install/bin/clang++ \
        ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --build-only \
        --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench
      srun -p i8cpu --time=02:00:00 --exclusive \
        ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --run-only \
        --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
        --threads "4 16 64" --xlat-cpu 0,1

Then: whether to move the mimalloc-bench pin to v1.2.0 now for the
use-after-free, or to fix these first and pin v1.2.1.

### 4.1 M5 Ultra (2026-10-10, 5 repetitions)

`kamepoolalloc/tests/bench/bench_ab.sh --threads "4 8"`, run from `~`
(work directory `~/bench-ab`); every run exited 0 with a result.  Not idle:
Chrome Remote Desktop's host used about two cores throughout and the KAME
app was open; load average 5.5 at the start, 4.1 at the end.  xthread at
w8 runs 16 threads on 10 P cores.

`provenance.txt`:

    date      2026-10-10T13:39:01Z
    host      dhcp-029-018.issp.u-tokyo.ac.jp
    kernel    Darwin 27.0.0 arm64
    cpu       Apple M5 Ultra, 10P + 20E, 96 GiB, Mac17,15
    os        macOS 27.0.1 (26A434)
    power     Now drawing from 'AC Power'
    benches   /Users/ssp/KAME/kamepoolalloc @ 01ced5831, /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ (Apple clang version 21.0.0 (clang-2100.3.34.2))
    arm       kp-v1.1.0: /Users/ssp/bench-ab/kp-v1.1.0/out/libkamepoolalloc.dylib (558280 B, sha256 72586741ef27e4af, commit a26213a, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    arm       kp-v1.2.0: /Users/ssp/bench-ab/kp-v1.2.0/out/libkamepoolalloc.dylib (697480 B, sha256 1ea7c91afbeda2c1, commit f9cd8ed, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    mbench    /Users/ssp/bench-ab/mimalloc-bench @ 69c41ed, binaries in /Users/ssp/bench-ab/mbench-build
    params    reps=5 threads="4 8" only="loop xthread xmalloc larson rptest xlat" quick=0 xlat="1024x200000 1024x1000000 3000x200000" xlat_cpu=none loop_iters=20000000

`summary.md`:

| bench | config | metric | kp-v1.1.0 | kp-v1.2.0 | kp-v1.2.0 / kp-v1.1.0 |
|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1138 (1133–1143) | 1129 (1108–1140) | 0.99× |
| loop | 1024B | Mops ↑ | 700 (699–701) | 674 (657–692) | 0.96× |
| loop | 16384B | Mops ↑ | 827 (826–832) | 758 (757–762) | 0.92× |
| xthread | w4,1KiB | Mfree ↑ | 37.3 (36.9–37.8) | 82.9 (80.6–87.0) | 2.22× |
| xmalloc | w4 | Mfree ↑ | 354 (297–362) | 375 (370–378) | 1.06× |
| larson | t4 | Mops ↑ | 138 (138–139) | 99.9 (99.4–100) | 0.72× |
| rptest | t4 | Mops_cpu ↑ | 4.29 (4.01–4.52) | 4.39 (4.18–4.63) | 1.02× |
| xthread | w8,1KiB | Mfree ↑ | 43.2 (43.0–44.2) | 122 (118–127) | 2.83× |
| xmalloc | w8 | Mfree ↑ | 491 (483–491) | 491 (489–494) | 1.00× |
| larson | t8 | Mops ↑ | 211 (209–213) | 151 (150–152) | 0.71× |
| rptest | t8 | Mops_cpu ↑ | 2.54 (2.39–2.68) | 2.56 (2.42–2.85) | 1.00× |
| xlat | 1024Bx200000 | rate_M ↑ | 0.014 (0.012–0.014) | 4.63 (4.57–4.7) | 331.07× |
| xlat | 1024Bx200000 | med_ns ↓ | 73542 (69792–81083) | 125 (125–125) | 0.00× |
| xlat | 1024Bx200000 | p999_ns ↓ | 115084 (104375–121125) | 292 (292–333) | 0.00× |
| xlat | 1024Bx200000 | max_ns ↓ | 226542 (182792–234458) | 15541 (14625–34375) | 0.07× |
| xlat | 1024Bx1000000 | rate_M ↑ | 0.002 (0.002–0.002) | 4.65 (4.63–4.68) | 2326.00× |
| xlat | 1024Bx1000000 | med_ns ↓ | 564959 (562958–567459) | 125 (125–125) | 0.00× |
| xlat | 1024Bx1000000 | p999_ns ↓ | 859875 (858458–864292) | 334 (292–334) | 0.00× |
| xlat | 1024Bx1000000 | max_ns ↓ | 1518333 (1041666–1649166) | 15042 (13791–16375) | 0.01× |
| xlat | 3000Bx200000 | rate_M ↑ | 0.006 (0.006–0.006) | 3.81 (3.75–3.85) | 635.50× |
| xlat | 3000Bx200000 | med_ns ↓ | 178084 (175375–179583) | 167 (167–167) | 0.00× |
| xlat | 3000Bx200000 | p999_ns ↓ | 295916 (288000–297542) | 334 (334–375) | 0.00× |
| xlat | 3000Bx200000 | max_ns ↓ | 368417 (368125–379083) | 14042 (13333–34250) | 0.04× |

## 5. Causes and the fix (2026-10-11, M5 Ultra)

Single-threaded, none of v1.2.0's cross-thread machinery runs (revival
stacks, groups), so its single-thread sequence of operations should have
been v1.1.0's.  It was not, in three places, and each was a regression:

1. **The FS=false free (§fl-avail).**  v1.1.0's free pushed the slot and,
   in the borrow tier, re-aimed the bucket's shortcut at its chunk
   (§freelist-follow) — one store.  §fl-avail kept per-bucket lists of
   chunks with freelist entries exact on every free: a free that re-aimed
   the shortcut away from a chunk with entries read that chunk's head (its
   cache line) and listed it.  On a random-victim workload that is nearly
   every free: larson 0.85x from this alone, a single-thread random churn
   (5000 blocks of 384..1024 B) 0.70x.  In the full-usable tier the test
   made `deallocate_fs_false_owner` 20 instructions from entry to `ret`;
   the M5 fetches 16 per cycle per straight-line run (padded with `nop`s,
   16 instructions ran at 825 M ops/s, 17 at 763, 18 or more at 754), hence
   bench_loop 16 KiB 0.92x.
2. **The miss (§fl-avail).**  v1.1.0 took the first chunk in DLL order with
   entries (§24); §fl-avail took the chunk listed last.
3. **Bitmap room in the thread's own chunks (§revive).**  §revive dropped
   v1.1.0's cursor walk of the DLL, keeping only chunks that cross-thread
   frees revived.  A variable-size chunk stops being the pin when a request
   finds no run of N free slots in it, yet keeps room for smaller ones;
   nothing found that room again.  20 K blocks of 8..1000 B replaced 4 times
   each, one thread: 64 chunks in v1.1.0, 266 in v1.2.0; larson's heap went
   from 3 regions to 13 and its RSS from ~100 MB to ~300 MB.

The fix restores v1.1.0's sequence and bounds only what made v1.1.0 slow:

- **§fl-scan.**  The free is v1.1.0's plus one `tbnz` (FS=true and both
  FS=false tiers).  A miss scans from the DLL head for the first chunk with
  entries, as v1.1.0 did, at most 64 chunks (beyond that, round a cursor),
  and not at all while the bucket is "dry": a scan that finds nothing marks
  every freelist but the shortcut's empty (FL_UNTRACKED, value 1; slots are
  at least 32-aligned), and a free onto a marked freelist sees bit 0 of the
  head it loads anyway and ends the dry spell.  That is what kept v1.1.0's
  allocate-only runs from going quadratic (1 KiB, 200 K live: 14 us per
  allocation in v1.1.0, 87 ns here and in v1.2.0).  The §fl-avail lists are
  gone.
- **§walk.**  v1.1.0's bitmap walk (cursor, exhausted flag) again, after the
  revival lists, at most 64 chunks per call.  The churn above: 64 chunks.
- Smaller: the lean free reads the page with `kame_page_or_null()` (no
  frame); `kame_malloc_slow` pops 369..32768 B frameless; the out-of-line
  hot entries are 64-byte aligned (`KAME_HOT_ALIGN`); a region-cache miss on
  a pool pointer refills the cache inline instead of going cold; an empty
  `asm volatile` keeps the full tier's `ret` from being merged with the
  borrow tier's (a taken branch to it cost 16 KiB a few percent).

Single-thread random churn (M ops/s; 8..1000 B / 16..368 B / 100 K blocks
of 8..1000 B): v1.1.0 73–78 / 78 / 5.0, v1.2.0 54 / – / 27.6, fix 78–79 /
78.6 / 36.3.  Allocate-then-refill (free every other of 200 K 1 KiB blocks,
allocate them back): v1.1.0 208 ns, v1.2.0 17 ns, fix 27 ns per allocation.

`bench_ab.sh --threads "4 8"`, 5 repetitions, with the fix as a third arm
(`--arm fix=...`, built from `claude/v1.2-perf` with the same recipe).  The
machine as in §4.1 (Chrome Remote Desktop's host using about two cores).
xthread w4 is the one row below v1.2.0 (0.97x, ranges nearly touching).

| bench | config | metric | kp-v1.1.0 | kp-v1.2.0 | fix | kp-v1.2.0 / kp-v1.1.0 | fix / kp-v1.1.0 |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1138 (1132–1148) | 1135 (1131–1147) | 1136 (1131–1148) | 1.00× | 1.00× |
| loop | 1024B | Mops ↑ | 701 (696–706) | 674 (656–701) | 798 (792–849) | 0.96× | 1.14× |
| loop | 16384B | Mops ↑ | 825 (823–836) | 757 (755–765) | 843 (821–883) | 0.92× | 1.02× |
| xthread | w4,1KiB | Mfree ↑ | 36.8 (36.3–38.2) | 81.7 (81.1–87.7) | 79.4 (78.1–81.0) | 2.22× | 2.16× |
| xmalloc | w4 | Mfree ↑ | 357 (349–359) | 377 (370–379) | 386 (373–393) | 1.06× | 1.08× |
| larson | t4 | Mops ↑ | 138 (136–141) | 99.6 (99.4–103) | 146 (142–151) | 0.72× | 1.06× |
| rptest | t4 | Mops_cpu ↑ | 4.38 (4.18–4.54) | 4.12 (4.08–4.77) | 4.42 (4.06–4.71) | 0.94× | 1.01× |
| xthread | w8,1KiB | Mfree ↑ | 43.7 (43.3–44.3) | 121 (120–126) | 120 (117–125) | 2.77× | 2.75× |
| xmalloc | w8 | Mfree ↑ | 495 (487–497) | 496 (487–498) | 506 (506–512) | 1.00× | 1.02× |
| larson | t8 | Mops ↑ | 211 (210–213) | 151 (148–152) | 225 (218–228) | 0.71× | 1.07× |
| rptest | t8 | Mops_cpu ↑ | 2.52 (2.49–2.72) | 2.6 (2.57–2.79) | 2.6 (2.47–2.69) | 1.03× | 1.03× |
| xlat | 1024Bx200000 | rate_M ↑ | 0.014 (0.013–0.015) | 4.64 (4.59–4.87) | 4.92 (4.8–4.98) | 331.71× | 351.07× |
| xlat | 1024Bx200000 | med_ns ↓ | 69625 (66250–76833) | 125 (125–125) | 125 (125–125) | 0.00× | 0.00× |
| xlat | 1024Bx200000 | p999_ns ↓ | 110833 (101459–114250) | 292 (292–459) | 292 (292–292) | 0.00× | 0.00× |
| xlat | 1024Bx200000 | max_ns ↓ | 197583 (185875–396542) | 19416 (13459–35250) | 17208 (13125–29916) | 0.10× | 0.09× |
| xlat | 1024Bx1000000 | rate_M ↑ | 0.002 (0.002–0.002) | 4.71 (4.66–4.86) | 4.9 (4.87–5.11) | 2352.50× | 2448.00× |
| xlat | 1024Bx1000000 | med_ns ↓ | 558084 (555500–563416) | 125 (83.0–125) | 125 (83.0–125) | 0.00× | 0.00× |
| xlat | 1024Bx1000000 | p999_ns ↓ | 780542 (763209–866500) | 333 (292–458) | 292 (292–334) | 0.00× | 0.00× |
| xlat | 1024Bx1000000 | max_ns ↓ | 937625 (791375–1665500) | 15416 (13917–22834) | 14167 (13541–36959) | 0.02× | 0.02× |
| xlat | 3000Bx200000 | rate_M ↑ | 0.006 (0.006–0.006) | 3.84 (3.76–3.95) | 4.03 (3.98–4.18) | 639.67× | 672.33× |
| xlat | 3000Bx200000 | med_ns ↓ | 169791 (165750–177708) | 167 (167–167) | 167 (166–167) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | p999_ns ↓ | 283125 (275042–296625) | 333 (333–334) | 292 (292–333) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | max_ns ↓ | 396542 (295875–414583) | 14292 (13042–15334) | 14250 (13333–17083) | 0.04× | 0.04× |

`provenance.txt`:

    date      2026-10-10T16:42:00Z
    host      dhcp-029-018.issp.u-tokyo.ac.jp
    kernel    Darwin 27.0.0 arm64
    cpu       Apple M5 Ultra, 10P + 20E, 96 GiB, Mac17,15
    os        macOS 27.0.1 (26A434)
    power     Now drawing from 'AC Power'
    benches   /Users/ssp/KAME/kamepoolalloc @ 34799c70d, /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ (Apple clang version 21.0.0 (clang-2100.3.34.2))
    arm       kp-v1.1.0: /Users/ssp/bench-ab/kp-v1.1.0/out/libkamepoolalloc.dylib (558280 B, sha256 72586741ef27e4af, commit a26213a, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    arm       kp-v1.2.0: /Users/ssp/bench-ab/kp-v1.2.0/out/libkamepoolalloc.dylib (697480 B, sha256 1ea7c91afbeda2c1, commit f9cd8ed, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    arm       fix: /Users/ssp/bench-ab/arms/final/out/libkamepoolalloc.dylib (685208 B, sha256 6f14cba89be3459c)
    mbench    /Users/ssp/bench-ab/mimalloc-bench @ 69c41ed, binaries in /Users/ssp/bench-ab/mbench-build
    params    reps=5 threads="4 8" only="loop xthread xmalloc larson rptest xlat" quick=0 xlat="1024x200000 1024x1000000 3000x200000" xlat_cpu=none loop_iters=20000000
