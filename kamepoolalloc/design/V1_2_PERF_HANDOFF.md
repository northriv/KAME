# Handoff: v1.2.0 against v1.1.0 — where it is faster, and where it is slower

Started 2026-10-10 on a MacBook Air M3.  The regressions reproduced on the
M5 Ultra (§4.1); their causes and the fix are in §5 (2026-10-11), which
measures at or above v1.1.0 on every workload here, and so it does on the M3
(§6).  Its Linux validation is green (`POOL_FL_AVAIL_LINUX_CHECK.md` §8:
no-DCAS 3/3, six trees, exclusivity 0/2000, soak, UBSan, no growth).
On Ohtaka (§7, x86-64) the fix brings larson back from 0.37–0.42x to
1.02–1.04x and rptest from 0.59–0.84x to 0.90–1.00x.
The macOS rptest rows (§3.1's smoke, §4.1, §5, §6) did not measure kame's
allocation: rptest allocates only through `calloc` and `posix_memalign`,
which the macOS dylib left to libsystem until `9bb57a41f`, so every arm
there was libsystem's allocator plus that arm's `free` forwarding.  The
Ohtaka rptest rows are kame's — Linux always interposed both.
**Open:** two Ohtaka rows below v1.1.0 — rptest t16 0.90x and xthread w64
0.70x (wide ranges; v1.2.0 0.81x) — the 0.2–0.8 ms xlat maximum there
against ~15 us on the M5, and the pin decision at the end of §4.  Append
results here.

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
        --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
        --tags v1.1.0,v1.2.0,tree
      # one wrapper script per job (srun's check_usage wrapper mangles
      # compound arguments), e.g. ~/kame-claude/bench-ab/jobs/run_A.sh:
      #   exec ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --run-only \
      #     --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
      #     --tags v1.1.0,v1.2.0,tree --threads "4 16 64" --xlat-cpu 0,1 \
      #     --only "loop xthread"
      for n in A B C D1 D2 D3; do
        srun -p i8cpu --time=00:30:00 --exclusive -c 128 ./run_$n.sh > $n.log 2>&1
      done

  with A `--only "loop xthread"`, B `"xmalloc larson"`, C `rptest`, and xlat
  split three ways: D1 `--xlat "1024x200000 3000x200000"`, D2 and D3
  `--xlat 1024x1000000` with `--reps 3` and `--reps 2`.  Run 2026-10-11: §7.

  Two traps in the single `srun --time=03:00:00` this section first gave:
  i8cpu's QOS caps a job at 30 minutes (and runs one job per user at a
  time), so the whole set has to be split, and v1.1.0 alone needs about 25
  minutes for five repetitions of xlat 1 KiB x 1 M; and **srun without `-c`
  gives the task one CPU** (`nproc` 1, provenance "1 cpus"), so every
  multi-thread workload runs on a single core.  `-c 128` gives it 0-127.

  (`tree` is the checkout's kamepoolalloc/, i.e. the fix of §5 on master.
  Before 2026-10-11 the default compared v1.1.0 with v1.2.0 and left the fix
  out.)

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

## 6. The fix on the M3 (2026-10-11)

`2e8f940fd` (master) built with the same recipe, Apple clang 17.0.0 on the
MacBook Air M3 of §3 (macOS 26.7, AC power), as a third arm.

The single-thread churn of §5 (`replace1`: 5000 live blocks, each iteration
frees a random one and allocates a replacement; M ops/s / chunks live at the
end; maximum RSS was v1.1.0's in every row):

| sizes | v1.1.0 | v1.2.0 | fix |
|---|---|---|---|
| 400..1000 B | 131 / 29 | 83 / 266 | 127 / 29 |
| 8..1000 B | 128 / 33 | 84 / 227 | 132 / 33 |
| 1500..3000 B | 86.5 / 39 | 74.8 / 300 | 86.5 / 39 |
| 700 B | 334 / 31 | 271 / 31 | 331 / 31 |
| 1024 B | 290 / 38 | 262 / 38 | 300 / 38 |

`bench_ab.sh --threads "2 4" --xlat "1024x200000 3000x200000"`, 5
repetitions (the 1 M live configuration left out: v1.1.0 takes over a
minute per run of it there):

| bench | config | metric | v1.1.0 | v1.2.0 | v1.2.1c | v1.2.0 / v1.1.0 | v1.2.1c / v1.1.0 |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 699 (414–752) | 746 (558–749) | 742 (599–760) | 1.07× | 1.06× |
| loop | 1024B | Mops ↑ | 499 (418–506) | 443 (441–450) | 537 (513–538) | 0.89× | 1.08× |
| loop | 16384B | Mops ↑ | 576 (422–580) | 502 (501–505) | 579 (546–580) | 0.87× | 1.01× |
| xthread | w2,1KiB | Mfree ↑ | 16.4 (15.7–16.5) | 47.6 (36.7–90.5) | 53.0 (35.1–86.5) | 2.90× | 3.23× |
| xmalloc | w2 | Mfree ↑ | 156 (154–157) | 153 (149–157) | 157 (155–161) | 0.98× | 1.01× |
| larson | t2 | Mops ↑ | 61.3 (59.9–61.5) | 41.7 (39.9–42.3) | 64.4 (62.2–65.6) | 0.68× | 1.05× |
| rptest | t2 | Mops_cpu ↑ | 4.2 (3.94–4.69) | 4.18 (3.87–4.94) | 4.1 (3.8–5.05) | 1.00× | 0.98× |
| xthread | w4,1KiB | Mfree ↑ | 20.2 (19.6–21.4) | 39.8 (35.8–43.9) | 39.1 (36.5–43.6) | 1.96× | 1.93× |
| xmalloc | w4 | Mfree ↑ | 267 (255–279) | 269 (261–272) | 271 (244–285) | 1.01× | 1.02× |
| larson | t4 | Mops ↑ | 105 (103–111) | 77.0 (75.0–80.7) | 120 (115–125) | 0.73× | 1.14× |
| rptest | t4 | Mops_cpu ↑ | 4.06 (3.37–4.17) | 4.03 (3.59–4.35) | 3.88 (3.39–4.33) | 0.99× | 0.96× |
| xlat | 1024Bx200000 | rate_M ↑ | 0.022 (0.02–0.025) | 6.51 (5.33–6.82) | 6.99 (5.77–7.11) | 296.05× | 317.55× |
| xlat | 1024Bx200000 | med_ns ↓ | 43333 (40042–49916) | 83.0 (42.0–125) | 42.0 (42.0–125) | 0.00× | 0.00× |
| xlat | 1024Bx200000 | p999_ns ↓ | 79416 (73958–259708) | 459 (250–833) | 333 (250–625) | 0.01× | 0.00× |
| xlat | 1024Bx200000 | max_ns ↓ | 144250 (113708–140448459) | 44667 (5334–136958) | 14459 (5167–40917) | 0.31× | 0.10× |
| xlat | 3000Bx200000 | rate_M ↑ | 0.007 (0.007–0.007) | 4.49 (3.97–6.13) | 5.33 (4.8–5.7) | 642.00× | 760.86× |
| xlat | 3000Bx200000 | med_ns ↓ | 135959 (133958–141959) | 125 (42.0–166) | 42.0 (42.0–84.0) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | p999_ns ↓ | 232959 (223250–832041) | 375 (292–1167) | 375 (250–959) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | max_ns ↓ | 341500 (279333–3636458) | 22792 (7625–146958) | 41209 (5667–123917) | 0.07× | 0.12× |

rptest is the one row under v1.1.0 (0.96-0.98x), within ranges that
overlap almost entirely; v1.2.0 sits at the same level.  The full ctest
tree passes, 43/43 (Release), and `tools/audit/run_audits.sh` is clean --
its no-DCAS i486 probes skip on macOS, so they are still to run on Linux.

Read for the FL_UNTRACKED marker: every free path that can push onto a
marked head -- the lean, off-lean and cold frees, FS=true and both FS=false
tiers -- tests bit 0 of the head it loaded and rewrites the slot's next to
nullptr before anything can pop it; the shortcut is only ever aimed by
`fl_retarget` (which clears a mark on the cell it aims at), by the
borrow-tier follow (the cell it just pushed onto) or at nullptr; and the
exit drain treats the marker as empty.  The §walk and §fl-scan cursors are
moved off a chunk the empty-neighbour release unlinks and reset at exit.

## 7. Ohtaka (2026-10-11)

`bench_ab.sh --tags v1.1.0,v1.2.0,tree --threads "4 16 64" --xlat-cpu 0,1`,
5 repetitions, split into six i8cpu jobs as in §4 (each job interleaves all
three arms on one node, so a row's ratios are within one node); every run
exited 0 with a result.  `tree` is `9f4cdc8be`, whose kamepoolalloc/ is the
fix of §5 (`2e8f940fd`) plus documentation and the bench script.  The 1 KiB
x 1 M xlat row merges D2 (3 repetitions) and D3 (2).  A first attempt
without `-c` ran on one CPU and is discarded.

| bench | config | metric | kp-v1.1.0 | kp-v1.2.0 | kp-tree | kp-v1.2.0 / kp-v1.1.0 | kp-tree / kp-v1.1.0 |
|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 261 (260–262) | 266 (265–266) | 262 (257–262) | 1.02× | 1.00× |
| loop | 1024B | Mops ↑ | 229 (228–229) | 247 (172–248) | 229 (228–229) | 1.08× | 1.00× |
| loop | 16384B | Mops ↑ | 172 (170–172) | 170 (169–171) | 173 (171–173) | 0.99× | 1.01× |
| xthread | w4,1KiB | Mfree ↑ | 9.64 (9.3–9.96) | 18.7 (16.9–19.2) | 18.5 (17.1–18.9) | 1.94× | 1.92× |
| xthread | w16,1KiB | Mfree ↑ | 40.0 (37.9–40.8) | 57.8 (52.6–67.5) | 59.3 (52.2–66.7) | 1.44× | 1.48× |
| xthread | w64,1KiB | Mfree ↑ | 79.8 (55.8–81.1) | 64.9 (51.1–72.1) | 56.1 (51.3–86.4) | 0.81× | 0.70× |
| xmalloc | w4 | Mfree ↑ | 122 (117–123) | 128 (125–129) | 129 (119–130) | 1.06× | 1.06× |
| larson | t4 | Mops ↑ | 57.2 (50.1–65.4) | 23.0 (21.8–24.0) | 58.7 (55.1–60.5) | 0.40× | 1.03× |
| xmalloc | w16 | Mfree ↑ | 469 (454–472) | 464 (456–508) | 493 (493–494) | 0.99× | 1.05× |
| larson | t16 | Mops ↑ | 216 (212–220) | 90.9 (86.9–92.6) | 221 (217–224) | 0.42× | 1.02× |
| xmalloc | w64 | Mfree ↑ | 732 (499–1307) | 1112 (635–1243) | 1140 (837–1377) | 1.52× | 1.56× |
| larson | t64 | Mops ↑ | 618 (615–650) | 226 (215–230) | 644 (638–656) | 0.37× | 1.04× |
| rptest | t4 | Mops_cpu ↑ | 7.2 (6.92–7.31) | 6.04 (5.84–6.24) | 7.22 (7.02–7.28) | 0.84× | 1.00× |
| rptest | t16 | Mops_cpu ↑ | 3.53 (2.96–3.61) | 2.5 (2.35–2.62) | 3.17 (3.01–3.49) | 0.71× | 0.90× |
| rptest | t64 | Mops_cpu ↑ | 1.09 (1.03–1.18) | 0.646 (0.616–0.747) | 1.04 (1–1.09) | 0.59× | 0.95× |
| xlat | 1024Bx200000 | rate_M ↑ | 0.005 (0.005–0.005) | 2.01 (2.01–2.13) | 2.1 (2.1–2.12) | 402.40× | 420.80× |
| xlat | 1024Bx200000 | med_ns ↓ | 190309 (189207–191471) | 421 (400–430) | 411 (411–411) | 0.00× | 0.00× |
| xlat | 1024Bx200000 | p999_ns ↓ | 260741 (259050–261923) | 1573 (1553–1593) | 1433 (1423–1453) | 0.01× | 0.01× |
| xlat | 1024Bx200000 | max_ns ↓ | 332949 (323881–354077) | 341395 (187714–460949) | 192894 (188315–196220) | 1.03× | 0.58× |
| xlat | 3000Bx200000 | rate_M ↑ | 0.003 (0.003–0.003) | 2.06 (1.93–2.11) | 2.16 (2.14–2.16) | 686.00× | 719.00× |
| xlat | 3000Bx200000 | med_ns ↓ | 388872 (387450–390105) | 441 (411–471) | 401 (401–401) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | p999_ns ↓ | 593188 (591765–596053) | 1232 (1212–1262) | 1163 (1162–1182) | 0.00× | 0.00× |
| xlat | 3000Bx200000 | max_ns ↓ | 652920 (614407–663549) | 611853 (194577–857661) | 404301 (194047–741557) | 0.94× | 0.62× |
| xlat | 1024Bx1000000 | rate_M ↑ | 0.001 (0.001–0.001) | 1.97 (1.94–1.99) | 2.04 (2.02–2.06) | 1966.00× | 2036.00× |
| xlat | 1024Bx1000000 | med_ns ↓ | 895237 (894085–910686) | 431 (431–441) | 421 (421–421) | 0.00× | 0.00× |
| xlat | 1024Bx1000000 | p999_ns ↓ | 1215261 (1208146–1221563) | 1703 (1523–1743) | 1603 (1393–1603) | 0.00× | 0.00× |
| xlat | 1024Bx1000000 | max_ns ↓ | 1257761 (1219970–1307425) | 445620 (197362–779900) | 524749 (188385–731899) | 0.35× | 0.42× |

- **larson** is v1.2.0's worst row on x86-64 — 0.37–0.42x, against
  0.67–0.74x on the M3 — and the fix takes it to 1.02–1.04x.
- **rptest**: v1.2.0 0.59–0.84x, getting worse with threads, which the Macs
  (at most 8 threads) did not show; the fix 0.90–1.00x.  t16 is the one
  below 0.95x, with ranges that overlap (v1.1.0 2.96–3.61, tree 3.01–3.49).
- **xthread w64** (64 producers + 64 consumers on 128 cores): 0.70x for the
  tree, 0.81x for v1.2.0, but all three arms span ~50–85 within themselves;
  not a result until it is repeated with more repetitions.  w4 and w16 are
  1.9x and 1.5x as on the Macs.
- **bench_loop**: no single-thread regression here even in v1.2.0 (1 KiB
  1.08x, 16 KiB 0.99x; one 172 outlier in v1.2.0's 1 KiB), so what §5's
  cause 1 cost the Macs (the extra free work, and the 16 KiB fetch-width
  step) does not show on Zen 2; the fix is at 1.00x.
- **xlat**: the quadratic v1.1.0 fill is 400–2000x off, as on the Macs; the
  fix is slightly ahead of v1.2.0 in rate (2.04–2.16 against 1.97–2.06 M/s)
  and p99.9.  The maximum of both is 0.2–0.8 ms, against ~15 us on the M5
  Ultra with no pinning — not looked into (THP is `always` on these nodes;
  the producer and consumer are pinned to CPUs 0 and 1).

Provenance (job B; the others differ in date, job, node and params):

    date      2026-10-11T08:41:25Z
    host      c15u01n2
    kernel    Linux 4.18.0-372.9.1.el8.x86_64 x86_64
    cpu       AMD EPYC 7702 64-Core Processor, 128 cpus, 8 NUMA nodes
    thp       [always] madvise never
    governor  n/a
    libc      ldd (GNU libc) 2.28
    slurm     job 3111785, partition i8cpu, node c15u01n2
    benches   /home/k0797/k079700/kame/kamepoolalloc @ 9f4cdc8be, /home/k0797/k079700/llvm-install/bin/clang++ (clang version 18.1.8)
    arm       kp-v1.1.0: /home/k0797/k079700/kame-claude/bench-ab/kp-v1.1.0/out/libkamepoolalloc.so (605504 B, sha256 254863f27cec6dac, commit a26213a, built by /home/k0797/k079700/llvm-install/bin/clang++)
    arm       kp-v1.2.0: /home/k0797/k079700/kame-claude/bench-ab/kp-v1.2.0/out/libkamepoolalloc.so (826480 B, sha256 c8ed4f197b189c84, commit f9cd8ed, built by /home/k0797/k079700/llvm-install/bin/clang++)
    arm       kp-tree: /home/k0797/k079700/kame-claude/bench-ab/kp-tree/out/libkamepoolalloc.so (806344 B, sha256 1bd4427d511261ca, tree /home/k0797/k079700/kame/kamepoolalloc at 9f4cdc8be, built by /home/k0797/k079700/llvm-install/bin/clang++)
    mbench    /home/k0797/k079700/mimalloc-bench @ 941c265, binaries in /home/k0797/k079700/mimalloc-bench/out/bench
    params    reps=5 threads="4 16 64" only="xmalloc larson" quick=0 xlat="1024x200000 1024x1000000 3000x200000" xlat_cpu=0,1 loop_iters=20000000

| job | slurm job | node | start (UTC) | `--only` / `--xlat` / reps |
|---|---|---|---|---|
| A | 3111782 | c15u01n1 | 08:37 | loop xthread |
| B | 3111785 | c15u01n2 | 08:41 | xmalloc larson |
| C | 3111789 | c15u01n1 | 08:52 | rptest |
| D1 | 3111793 | c15u01n1 | 09:10 | xlat, 1024x200000 3000x200000 |
| D2 | 3111795 | c15u01n2 | 09:15 | xlat, 1024x1000000, 3 reps |
| D3 | 3111807 | c15u01n1 | 09:31 | xlat, 1024x1000000, 2 reps |

The results directories are `~/kame-claude/bench-ab/results-*` on Ohtaka;
the job logs, wrappers and the merged `summary-all.md` are in
`~/kame-claude/bench-ab/jobs/`.
