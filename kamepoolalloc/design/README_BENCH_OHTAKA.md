# The README's Ohtaka benchmarks, re-measured for v1.2.1 (2026-10-12)

Re-measurement of every Ohtaka table in `kamepoolalloc/README.md` with the
v1.2.1 candidate — kamepoolalloc as of KAME `19b2e4758` (the runs report
`62e99d527`, which adds only an earlier draft of this note on top).  The
README's tables are from June, before v1.1.0:

| README table | old kame | re-measured in |
|---|---|---|
| `bench_compare.sh` 1T / 4 processes / 128 processes (the three Ohtaka charts) | `efbe6dcb` (allocator `5e127eb5`) | §1 |
| `alloc_tune_report` | `0e9413a6` | §2 |
| mimalloc-bench suite: xmalloc-test, mstress, rptest (+ larson, new) | `9ecc613d` | §3 |

The README's Ohtaka tables, charts and their text were replaced from §1–§3
on 2026-10-12, and again from §5 — every table re-measured at `e0974e402`,
with interleaved checks against v1.2.1 — on 2026-10-14.

## Common setup

- **Partition / nodes**: Ohtaka `i8cpu`, AMD EPYC 7702 (Zen 2), 2 × 64 cores
  = 128 cpus, 1 thread/core, 8 NUMA nodes; kernel
  `4.18.0-372.9.1.el8.x86_64`, glibc 2.28; THP `[always]`; no cpufreq
  governor or boost control exposed (`scaling_governor` and `cpufreq/boost`
  absent), `CPU MHz` 1996.  Every job: `sbatch -p i8cpu -N 1 -c 128
  --exclusive --time=00:30:00` (`nproc` = 128 inside each), one workload per
  job, each job's log prints `lscpu`, `lscpu --extended`, `numactl
  --hardware`, `dmesg | tail` (only `drop_caches` lines — the epilogue of the
  previous job — except one segfault `Code:` line before mstress, left by
  larson's t128 crashes in the job before it on the same node, §3.2) and the
  load average at start (a decaying
  remnant of the previous job; the node is exclusive).  The job script is
  `~/kame-claude/readme-v121/job.sbatch`; its parameters go by environment
  because `sbatch` splits quoted script arguments.
- **Compiler**: clang 18.1.8 (`~/llvm-install`) throughout.
- **Comparators**: `~/mimalloc-bench` at `941c265`, the builds the June
  tables used — mi3 = mimalloc v3.3.2, mi = v1.8.2, je = jemalloc 5.3.1,
  tc = gperftools 2.18 `tcmalloc_minimal`; sys = glibc 2.28.  In
  `bench_compare.sh` "mimalloc" is mi3 and "jemalloc" je, as in June.

| job | SLURM | node | JST (10-12) | what |
|---|---|---|---|---|
| bc4 | 3112115 | c15u02n1 | 01:06–01:08 | `bench_compare.sh --threads 4` |
| bc128 | 3112116 | c15u01n1 | 01:09–01:12 | `bench_compare.sh --threads 128` |
| tune | 3112117 | c15u04n4 | 01:08–01:09 | `alloc_tune_report` |
| xm64 | 3112118 | c15u01n1 | 01:12–01:25 | `bench_ab.sh --only xmalloc --xmalloc-sizes 64` |
| xm1024 | 3112119 | c15u01n1 | 01:25–01:38 | `bench_ab.sh --only xmalloc --xmalloc-sizes 1024` |
| larson | 3112120 | c15u01n2 | 01:38–01:54 | `bench_ab.sh --only larson` |
| mstress | 3112122 | c15u01n2 | 01:54–02:04 | `bench_ab.sh --only mstress` |
| rpt-1-4 | 3112128 | c15u01n2 | 02:04–02:20 | `bench_ab.sh --only rptest --threads "1 4"` |
| rpt-16-64 | 3112136 | c15u01n2 | 02:20–02:36 | `bench_ab.sh --only rptest --threads "16 64"` |
| rpt-128 | 3112144 | c15u01n2 | 02:36–02:45 | `bench_ab.sh --only rptest --threads 128` |
| tune3 | 3112184 | c15u01n2 | 02:45–02:48 | `alloc_tune_report` × 3 (added to judge §2's noise) |

All jobs exited 0.  Apart from larson's t128 (§3.2), every run reported
`ok`.  rpt-16-64, rpt-128 and tune3 report the tree as `62e99d527-dirty`:
the only change was the uncommitted draft of this file
(`kamepoolalloc/design/`); `--run-only` does not rebuild, and the kame
library's sha256 is the same in every job.

## 1. `bench_compare.sh` — the 1T / 4-process / 128-process tables

Build tree `~/kame/build/tests` (configured from `tests/`, clang,
`CMAKE_CXX_FLAGS="-O3 -DNDEBUG"`, `CMAKE_CXX_FLAGS_RELWITHDEBINFO` empty),
rebuilt on the login node at this commit (`cmake --build . --target
kamepoolalloc bench_loop alloc_tune_report`):

- `kamepoolalloc-tests/libkamepoolalloc.so` is **806384 B, not the ~0.6 MB
  CLAUDE.md gives** — and still a plain `-O3` build: `flags.make` has `-O3
  -DNDEBUG` and no `-g`, `readelf -S` lists no `.debug*` section, and its
  `size` (text 658064) matches bench_ab.sh's separate Release build (text
  657948).  The 0.2 MB is code added since June (598048 B on 06-16); the
  `-O2 -g` failure mode would be ≈ 2.1 MB.  The sanity figure in CLAUDE.md
  should read ≈ 0.8 MB.
- sha256 `4c7f914ce7add18f…`.

Run in `~/kame/build/tests`:

    ../../kamepoolalloc/tests/bench/bench_compare.sh --build-dir kamepoolalloc-tests \
      --mimalloc ~/mimalloc-bench/extern/mi3/out/release/libmimalloc.so \
      --jemalloc ~/mimalloc-bench/extern/je/lib/libjemalloc.so --threads 4   # and --threads 128

Both jobs print the 1T table.  The June tables were all measured on
`c15u01n1`; bc128 ran there again, bc4 landed on `c15u02n1`.

### 1T (median of 5, M ops/s) — bc128, `c15u01n1`; June value in brackets

| size   | system    | mimalloc  | jemalloc | kame          |
|--------|-----------|-----------|----------|---------------|
| 64B    | 212 (212) | **327** (331) | 182 (182) | 259 (260) |
| 1.0KB  | 210 (214) | 217 (203) | 160 (160) | **228** (236) |
| 16KB   | 69 (69)   | 95 (93)   | 45 (45)  | **173** (168) |
| 64KB   | 65 (65)   | **95** (95) | 4 (4)  | 79 (82)       |
| 256KB  | 71 (70)   | **95** (95) | 4 (4)  | 80 (87)       |
| 1.0MB  | 71 (71)   | 5 (5)     | 4 (4)    | **83** (84)   |
| 4.0MB  | 70 (67)   | 4 (4)     | 4 (4)    | 59 (62)       |

The same table from bc4 (`c15u02n1`):

| size      |  system | mimalloc | jemalloc |     kame |
|-----------|---------|----------|----------|----------|
| 64B       |     207 |      324 |      177 |      253 |
| 1.0KB     |     207 |      203 |      155 |      222 |
| 16KB      |      67 |       90 |       44 |      168 |
| 64KB      |      64 |       93 |        4 |       74 |
| 256KB     |      69 |       90 |        4 |       78 |
| 1.0MB     |      68 |        5 |        4 |       80 |
| 4.0MB     |      66 |        4 |        4 |       58 |

### 4 processes (aggregate, M ops/s, median of 5) — bc4, `c15u02n1`; June (`c15u01n1`) in brackets

| size   | system    | mimalloc    | jemalloc | kame          |
|--------|-----------|-------------|----------|---------------|
| 64B    | 825 (845) | **1245** (1261) | 711 (727) | 1014 (1045) |
| 16KB   | 268 (274) | 371 (381)   | 172 (177) | **681** (666) |
| 64KB   | 253 (259) | **370** (371) | 16 (18) | 308 (321)     |
| 1.0MB  | 277 (277) | 20 (20)     | 16 (17)  | **326** (335) |

### 128 processes (aggregate, M ops/s, median of 5) — bc128, `c15u01n1`; June in brackets

| size   | system        | mimalloc          | jemalloc      | kame              |
|--------|---------------|-------------------|---------------|-------------------|
| 64B    | 22575 (22634) | **39396** (39572) | 21184 (21280) | 31006 (29044)     |
| 16KB   | 7995 (7920)   | 11772 (11808)     | 5263 (5282)   | **22001** (21496) |
| 64KB   | 8101 (8128)   | **11864** (11866) | 469 (465)     | 9903 (10352)      |
| 1.0MB  | 8917 (8908)   | 562 (566)         | 473 (475)     | **10567** (10703) |

The comparators reproduce June to ±2 % on the same node, with two
exceptions (mimalloc 1 KiB 1T +7 %, system 4 MiB 1T +4 %).  kame is where it
was, to within a few percent: +7 % at 64 B / 128 processes (31006), +3 % at
16 KiB 1T, and −3 to −8 % at 1 KiB, 64 KiB, 256 KiB and 4 MiB 1T (228, 79,
80, 59) and at 64 KiB / 128 processes (9903, −4 %).  Every ordering in the
README holds: mimalloc leads at 64 B and 64–256 KiB, kame at 1 KiB, 16 KiB
and 1 MiB; at 4 MiB 1T the system allocator leads (70 vs 59; June 67 vs 62,
though the README bolds kame there).  On `c15u02n1` every column is 1–7 %
below `c15u01n1`, comparators included — node, not code.

## 2. `alloc_tune_report`

    cd ~/kame/build/tests && ./kamepoolalloc-tests/alloc_tune_report     # 2 s per bench (default)

Same library (sha256 `4c7f914ce7add18f…`, linked, not preloaded);
`alloc_tune_report` sha256 `eb75bc31774ad8cc…`.  Job tune, node `c15u04n4`.

Aggregate M ops/s (alloc → touch byte 0 → free); June (`0e9413a6`) in
brackets:

| size,tier                  | 1T        | 4T        | 16T         | 64T         | 128T              |
| -------------------------- | --------: | --------: | ----------: | ----------: | ----------------: |
| 64 B  (bucket)             | 132 (121) | 507 (482) | 2097 (1887) | 7656 (6356) | **12381** (10596) |
| 1 KiB (bucket)             | 114 (95)  | 457 (389) | 1824 (1543) | 6641 (5713) | **11253** (7831)  |
| 64 KiB (chunk)             | 43 (42)   | 31 (27)   | 69 (116)    | 171 (195)   | **353** (292)     |
| 1 MiB (chunk)              | 42 (44)   | 37 (31)   | 127 (81)    | 135 (132)   | **351** (543)     |
| 8 MiB (large_va)           | 29 (27)   | 7.4 (3)   | 4.4 (3)     | 3.6 (3)     | 2.8 (3)           |
| 40 MiB (large_va, cached)  | 12 (13)   | 2.6 (3)   | 3.7 (3)     | 2.2 (2)     | 2.5 (2)           |

The tier rows are one run each; tune3 repeated the report three times on one
node (`c15u01n2`) to see how far a cell moves (M ops/s, run 1 / 2 / 3):

| size,tier        | 16T               | 64T               | 128T                   |
| ---------------- | ----------------: | ----------------: | ---------------------: |
| 64 B  (bucket)   | 2094 / 2096 / 2094 | 7628 / 7457 / 7898 | 12488 / 11580 / 13924 |
| 1 KiB (bucket)   | 1822 / 1825 / 1826 | 6618 / 6379 / 6545 | 10350 / 10200 / 11429 |
| 64 KiB (chunk)   | 72 / 61 / 75      | 174 / 204 / 181   | 186 / 283 / 228        |
| 1 MiB (chunk)    | 216 / 172 / 218   | 237 / 148 / 179   | 345 / 426 / 431        |

The bucket tier is steady (±10 %) and clearly above June at every thread
count: 64 B 11.6–13.9 G ops/s at 128T (June 10.6), 1 KiB 10.2–11.4 G (June
7.8).  The chunk tier from 16 threads is not: a cell moves by up to 1.9×
between runs (64 KiB / 128T 186–353 over the four runs).  **1 MiB / 128T:
the four runs give 345–431, all below June's 543** — a possible regression
of the 1 MiB chunk tier at 128 threads, but June's figure is a single run
of a cell this noisy, so it is not established.  A same-node interleaved
A/B of the June library against this one on that loop would settle it
(`bench_ab.sh` has no chunk-tier multi-thread workload yet).  The 1-thread `munmap` baseline also
moved (69.9 / 175.8 / 170.8 µs), so the TLB-shootdown ratio printed at 128
threads ranges 8.7–21.6× with the absolute cost steady (1508–1696 µs).


Self-validation, as in June:

- `LRC_LAZY_INTERVAL_NS` 10 ms: 0.63 % worst-case per-thread wallclock —
  "default 10 ms is fine on this host; auto-tune kept it (raise-only)" (June:
  0.63 %, same message).
- `LRC_K_MAX` 256: "128 cores: K_MAX=256 (default) is appropriate".
- `MADV_HUGEPAGE`: 0.24 µs/page, the same as plain first-touch (0.24) — THP
  `always` already promotes.
- TLB shootdown, 32 MiB `munmap`: 69 µs at 1 thread, 1383 µs = **19.95×** at
  128 (June: 1543 µs, 22.01×); 6.2× at 4, 18.5× at 16, 16.8× at 64.

Full output of the tune job:

```
=== kamepoolalloc tuning report ===
Platform : Linux
CPUs     : 128
NUMA     : 8 nodes
Page sz  : 4096 B
THP      : always
Build    : LRC_K_MAX=256  LRC_N_MAX=40  LRC_HI=256 MiB
           (defaults — override with -DLRC_K_MAX=... etc.)

=== Raw VMM cost (32 MiB regions) ===
  mmap (no touch)                 0.5 us
  first-touch fault              0.24 us/page  (1.95 ms / 32 MiB)
  munmap (after touch)           63.5 us
  madvise(MADV_DONTNEED)         58.5 us
  mmap + MADV_HUGEPAGE touch     0.24 us/page  (1.99 ms / 32 MiB)

=== MT munmap contention (32 MiB, TLB shootdown effect) ===
    1 threads     69.3 us  (1.00x baseline)
    4 threads    427.8 us  (6.17x)
   16 threads   1281.5 us  (18.48x)
   64 threads   1163.8 us  (16.79x)
  128 threads   1383.4 us  (19.95x)

=== kamepoolalloc throughput (alloc+free, first-touch byte 0) ===
size,tier                  T        ns/op        M ops/s
64 B (bucket)              1            8         131.75
1 KiB (bucket)             1            9         114.29
64 KiB (chunk)             1           23          43.33
1 MiB (chunk)              1           24          42.49
8 MiB (large_va)           1           35          28.86
40 MiB (large_va, cached)     1           80          12.46

64 B (bucket)              4            2         507.01
1 KiB (bucket)             4            2         456.77
64 KiB (chunk)             4           32          31.07
1 MiB (chunk)              4           27          36.78
8 MiB (large_va)           4          136           7.36
40 MiB (large_va, cached)     4          383           2.61

64 B (bucket)             16            0        2096.71
1 KiB (bucket)            16            1        1823.84
64 KiB (chunk)            16           15          68.67
1 MiB (chunk)             16            8         126.96
8 MiB (large_va)          16          228           4.39
40 MiB (large_va, cached)    16          268           3.73

64 B (bucket)             64            0        7656.46
1 KiB (bucket)            64            0        6640.81
64 KiB (chunk)            64            6         171.11
1 MiB (chunk)             64            7         135.26
8 MiB (large_va)          64          280           3.58
40 MiB (large_va, cached)    64          453           2.21

64 B (bucket)            128            0       12381.18
1 KiB (bucket)           128            0       11252.57
64 KiB (chunk)           128            3         353.28
1 MiB (chunk)            128            3         350.79
8 MiB (large_va)         128          353           2.83
40 MiB (large_va, cached)   128          397           2.52

=== Recommendations ===

* §28.1 lazy drain:
    library auto-tuned interval   : 10 ms
    worst-case @ default 10 ms    : 0.63 % per-thread wallclock
    worst-case @ auto-tuned 10 ms : 0.63 % per-thread wallclock
  → default 10 ms is fine on this host; auto-tune kept it (raise-only).

* LRC_K_MAX guideline (current default 256):
  → 128 cores: K_MAX=256 (default) is appropriate

* NUMA: 8 nodes detected.  The slot array lives at one NUMA domain;
  cross-domain accesses pay remote-line latency.  Per-NUMA arenas are
  not implemented — workloads with strong NUMA locality may benefit.

=== end of report
```

## 3. mimalloc-bench suite (`bench_ab.sh`)

### 3.0 Method

Build (login node), then one job per workload:

    CC=~/llvm-install/bin/clang CXX=~/llvm-install/bin/clang++ \
      ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --build-only \
      --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
      --tags tree --only "xmalloc larson mstress rptest"

    X=~/mimalloc-bench/extern
    bench_ab.sh --run-only --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench --tags tree \
      --arm sys= --arm mi3=$X/mi3/out/release/libmimalloc.so --arm mi=$X/mi/out/release/libmimalloc.so \
      --arm je=$X/je/lib/libjemalloc.so --arm tc=$X/tc/.libs/libtcmalloc_minimal.so \
      --threads "1 4 16 64 128" --reps 5 --only <workload> [--xmalloc-sizes 64|1024]

One binary per workload, the arm switched by `LD_PRELOAD`; each repetition
runs all six arms back to back, the order alternating between repetitions;
median (min–max) of 5.  The kame arm is this checkout's kamepoolalloc/ built
by CMake Release (`-O3 -DNDEBUG`), `libkamepoolalloc.so` 806384 B, sha256
`e4603b3a2426fc38` — a separate build from §1's (same source and compiler,
different build dir).  Common `provenance.txt` lines (the per-job `date`,
`host`, `slurm`, `params` lines are in the job table and headings):

    kernel    Linux 4.18.0-372.9.1.el8.x86_64 x86_64
    cpu       AMD EPYC 7702 64-Core Processor, 128 cpus, 8 NUMA nodes
    thp       [always] madvise never
    governor  n/a
    libc      ldd (GNU libc) 2.28
    benches   /home/k0797/k079700/kame/kamepoolalloc @ 62e99d527, /home/k0797/k079700/llvm-install/bin/clang++ (clang version 18.1.8)
    arm       kp-tree: /home/k0797/k079700/kame-claude/bench-ab/kp-tree/out/libkamepoolalloc.so (806384 B, sha256 e4603b3a2426fc38, tree /home/k0797/k079700/kame/kamepoolalloc at 62e99d527, built by /home/k0797/k079700/llvm-install/bin/clang++)
    arm       sys: no preload (system malloc)
    arm       mi3: /home/k0797/k079700/mimalloc-bench/extern/mi3/out/release/libmimalloc.so (258848 B, sha256 e855d853149e3dc1)
    arm       mi: /home/k0797/k079700/mimalloc-bench/extern/mi/out/release/libmimalloc.so (142144 B, sha256 4ec607fe357d8afe)
    arm       je: /home/k0797/k079700/mimalloc-bench/extern/je/lib/libjemalloc.so (2452440 B, sha256 db00810fb6ebe2ba)
    arm       tc: /home/k0797/k079700/mimalloc-bench/extern/tc/.libs/libtcmalloc_minimal.so (200416 B, sha256 11fa25e465eeaf88)
    mbench    /home/k0797/k079700/mimalloc-bench @ 941c265, binaries in /home/k0797/k079700/mimalloc-bench/out/bench

Result directories (under `~/kame-claude/bench-ab/`, each with `raw.tsv`,
`summary.md`, `provenance.txt`, `logs/`): xm64
`results-c15u01n1-20261012-011256`, xm1024
`results-c15u01n1-20261012-012550`, larson
`results-c15u01n2-20261012-013853`, mstress
`results-c15u01n2-20261012-015419`, rptest
`results-c15u01n2-20261012-020412` (t1, t4), `…-022028` (t16, t64),
`…-023647` (t128).

### 3.1 xmalloc-test — `xmalloc-test -w N -t 5 -s S`, M frees/s ↑

N allocating threads + N freeing threads.  **The June table is 1024 B under
a 64 B title**: its command has no `-s`, xmalloc-test's default object size
is 1024 B (mimalloc-bench's `bench.sh` passes `-s 64`), and its mi3 column
(25.7 / 48.5 / 119 / 206 / 201) is today's 1024 B mi3 column (24.0 / 54.0 /
129 / 199 / 199), not the 64 B one (33.9 / 109 / 335 / 431 / 415).

**64 B** (`-s 64`, as mimalloc-bench runs it) — xm64, `c15u01n1`, 01:12:56:

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,64B | Mfree ↑ | 31.9 (31.2–32.4) | 5.63 (5.49–5.96) | 33.9 (33.3–35.5) | 34.7 (25.8–34.9) | 26.8 (21.1–28.6) | 8.88 (5.34–8.91) | 0.18× | 1.06× | 1.09× | 0.84× | 0.28× |
| xmalloc | w4,64B | Mfree ↑ | 127 (125–130) | 16.5 (15.7–17.5) | 109 (103–111) | 96.0 (91.7–99.2) | 92.7 (87.2–98.6) | 1.65 (1.52–1.86) | 0.13× | 0.86× | 0.75× | 0.73× | 0.01× |
| xmalloc | w16,64B | Mfree ↑ | 494 (490–499) | 67.9 (58.2–69.3) | 335 (303–361) | 248 (181–306) | 394 (325–406) | 1.1 (1.03–1.19) | 0.14× | 0.68× | 0.50× | 0.80× | 0.00× |
| xmalloc | w64,64B | Mfree ↑ | 1149 (709–1283) | 189 (180–194) | 431 (354–469) | 365 (347–406) | 867 (850–930) | 0.783 (0.771–0.799) | 0.16× | 0.38× | 0.32× | 0.75× | 0.00× |
| xmalloc | w128,64B | Mfree ↑ | 422 (333–539) | 186 (182–188) | 415 (393–456) | 391 (320–413) | 922 (874–965) | 0.843 (0.833–0.851) | 0.44× | 0.98× | 0.93× | 2.19× | 0.00× |

kame leads from 4 to 64 worker pairs (best other / kame 0.86×, 0.80×, 0.75×),
is level at 1 (mi 1.09×) — and at 128 falls from 1149 to 422, behind je
(922, 2.19×) and level with mi3 / mi.  No other arm drops like that from 64 to
128, and kame's w64 range (709–1283) is the widest in the table: the w128
step is kame's open item at 64 B.

**1024 B** (`-s 1024`, the default — what June ran) — xm1024, `c15u01n1`,
01:25:50; June kame in the last column:

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | June kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,1024B | Mfree ↑ | 6.17 (5.88–6.21) | 3.31 (3.08–3.53) | 24.0 (21.6–26.8) | 22.9 (22.2–23.1) | 5 (3.66–5.04) | 3.14 (2.9–3.25) | 0.54× | 3.89× | 3.72× | 0.81× | 0.51× | 2.9 |
| xmalloc | w4,1024B | Mfree ↑ | 18.3 (17.2–19.4) | 9.46 (8.69–9.65) | 54.0 (51.3–57.0) | 63.5 (60.0–65.6) | 15.5 (14.7–15.7) | 1.13 (0.969–1.15) | 0.52× | 2.95× | 3.48× | 0.85× | 0.06× | 8.8 |
| xmalloc | w16,1024B | Mfree ↑ | 71.7 (59.8–74.2) | 17.3 (15.7–20.4) | 129 (110–139) | 188 (165–205) | 55.8 (53.9–59.6) | 0.796 (0.74–0.874) | 0.24× | 1.79× | 2.62× | 0.78× | 0.01× | 35.8 |
| xmalloc | w64,1024B | Mfree ↑ | 51.1 (47.2–81.6) | 13.5 (12.0–19.5) | 199 (180–206) | 258 (240–271) | 173 (169–183) | 0.623 (0.6–0.634) | 0.26× | 3.89× | 5.04× | 3.38× | 0.01× | 60.8 |
| xmalloc | w128,1024B | Mfree ↑ | 76.2 (61.0–87.5) | 8.4 (7.79–8.79) | 199 (177–210) | 177 (169–189) | 160 (158–161) | 0.604 (0.592–0.606) | 0.11× | 2.62× | 2.33× | 2.10× | 0.01× | 47.3 |

Against June, kame is 2.1× faster at 1–4 pairs, 2.0× at 16, 1.6× at 128 —
and 16 % slower at 64 (51.1, range 47.2–81.6, vs 60.8).  The comparators
match June.  The gap to the leader shrank from 4.2–8.9× to 2.6–5.0×, but
1024 B cross-thread handoff remains kame's weakest workload: mi / mi3 lead at
every count and je from 64 pairs; kame beats glibc everywhere (1.9–9×) and
je up to 16 pairs.  w64 is again kame's noisiest and least-improved cell.

An earlier pass at `327644452` (2026-10-11, `c15u01n1`, same method)
agrees with both xmalloc tables to within run-to-run noise.

### 3.2 larson — `larson 5 8 1000 5000 100 4141 N`, M ops/s ↑ (new; the M5 note has it)

larson, `c15u01n2`, 01:38:53:

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| larson | t1 | Mops ↑ | 22.9 (22.7–23.1) | 21.8 (21.4–21.9) | 32.4 (32.0–32.6) | 42.0 (41.6–42.5) | 36.1 (35.9–36.3) | 45.1 (44.9–45.2) | 0.95× | 1.41× | 1.83× | 1.57× | 1.97× |
| larson | t4 | Mops ↑ | 62.8 (56.9–66.1) | 50.4 (47.1–53.2) | 63.7 (53.3–65.1) | 105 (87.0–113) | 93.1 (84.1–104) | 94.7 (87.9–104) | 0.80× | 1.02× | 1.67× | 1.48× | 1.51× |
| larson | t16 | Mops ↑ | 224 (223–227) | 170 (162–174) | 228 (219–234) | 370 (369–378) | 349 (342–351) | 336 (332–344) | 0.76× | 1.02× | 1.65× | 1.55× | 1.49× |
| larson | t64 | Mops ↑ | 650 (618–664) | 589 (578–596) | 878 (873–883) | 1335 (1313–1346) | 1260 (1243–1273) | 1101 (1090–1112) | 0.91× | 1.35× | 2.05× | 1.94× | 1.69× |

**No t128 row: larson cannot run 128 threads.**  Every arm, `sys`
included, dies with SIGSEGV (`exit=139`, all 30 runs) — larson.cpp has
`#define MAX_THREADS 100` and indexes `thread_data de_area[MAX_THREADS]` by
thread number.  mimalloc-bench's `bench.sh` would hit the same on a
128-cpu machine.

kame is ahead of glibc only (1.05–1.32×), level with mi3 at 4–16 threads,
and behind mi / je / tc by 1.5–2.1× at every count.  larson is the
worker-respawn pattern (threads exit and are replaced, their live blocks
freed by successors), so thread exit / adoption cost is a likely suspect;
not profiled here.

### 3.3 mstress — `mstress N 50 50`, wall s ↓

(mimalloc-bench's `bench.sh` runs `mstress N 50 25`; `50 50` is the
README's and `bench_ab.sh`'s.)  mstress, `c15u01n2`, 01:54:19; June kame in
the last column:

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | June kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mstress | t1 | wall_s ↓ | 0.0573 (0.0569–0.0732) | 0.0691 (0.069–0.0705) | 0.054 (0.0533–0.0552) | 0.0544 (0.053–0.0551) | 0.0596 (0.0586–0.0601) | 0.0569 (0.0558–0.0576) | 1.21× | 0.94× | 0.95× | 1.04× | 0.99× | 0.083 |
| mstress | t4 | wall_s ↓ | 0.603 (0.602–0.618) | 0.639 (0.614–0.659) | 0.448 (0.439–0.463) | 0.444 (0.426–0.452) | 0.929 (0.905–0.957) | 0.656 (0.645–0.665) | 1.06× | 0.74× | 0.74× | 1.54× | 1.09× | 0.620 |
| mstress | t16 | wall_s ↓ | 2.42 (2.37–2.43) | 2.64 (2.61–2.77) | 1.84 (1.77–1.88) | 1.91 (1.89–2.01) | 2.78 (2.72–3.15) | 2.34 (2.26–2.5) | 1.09× | 0.76× | 0.79× | 1.15× | 0.96× | 2.62 |
| mstress | t64 | wall_s ↓ | 6.3 (5.42–6.64) | 7.98 (7.72–8.14) | 4.03 (3.63–4.35) | 3.88 (3.8–4.04) | 8 (7.84–8.2) | 6.32 (5.94–6.52) | 1.27× | 0.64× | 0.62× | 1.27× | 1.00× | 6.22 |
| mstress | t128 | wall_s ↓ | 9.61 (8.53–9.91) | 14.4 (14.3–14.6) | 5.95 (5.91–6.21) | 5.87 (5.69–5.96) | 14.7 (14.3–15.0) | 10.2 (9.85–10.7) | 1.50× | 0.62× | 0.61× | 1.53× | 1.06× | 9.91 |

kame now beats glibc and jemalloc at every thread count (June: it trailed
glibc at 1–16), is within 4 % of or ahead of tcmalloc, and trails mi / mi3 by
1.3–1.6× from 4 threads up.  Against June's kame: 31 % faster at 1 thread,
3–8 % at 4, 16 and 128, equal at 64.  This node (`c15u01n2`) runs every arm
a few percent slower than the 10-11 pass on `c15u01n1` (sys 14.4 vs 14.0 s,
mi 5.87 vs 4.84 s at 128; kame 9.61 vs 8.44 s), so the June comparison,
made on an unrecorded node, carries that much uncertainty.

### 3.4 rptest — `rptest N 0 1 2 500 1000 100 8 16000`, M ops/CPU-s ↑

rpt-1-4, rpt-16-64, rpt-128, all `c15u01n2`; June kame (`9ecc613d`) in the
last column:

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | June kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rptest | t1 | Mops_cpu ↑ | 14.6 (14.4–15.3) | 10.9 (10.4–11.1) | 15.6 (15.4–16.0) | 15.6 (14.6–16.5) | 16.0 (15.9–16.3) | 17.9 (17.1–18.7) | 0.75× | 1.07× | 1.06× | 1.10× | 1.22× | 11.5 |
| rptest | t4 | Mops_cpu ↑ | 6.93 (6.62–7.11) | 2.75 (2.4–2.99) | 7.88 (6.25–8.24) | 9.69 (8.95–10.3) | 5.84 (5.38–6.42) | 6.92 (6.28–7.02) | 0.40× | 1.14× | 1.40× | 0.84× | 1.00× | 3.96 |
| rptest | t16 | Mops_cpu ↑ | 3.09 (2.71–3.75) | 2.13 (2.01–2.27) | 4.7 (3.31–4.84) | 5.98 (5.6–6.49) | 2.5 (2.37–2.57) | 2.27 (1.85–2.28) | 0.69× | 1.52× | 1.93× | 0.81× | 0.73× | 2.31 |
| rptest | t64 | Mops_cpu ↑ | 1.08 (0.957–1.09) | 0.56 (0.464–0.663) | 1.89 (1.48–2.2) | 2.51 (2.25–2.79) | 0.792 (0.702–1.12) | 0.222 (0.201–0.27) | 0.52× | 1.75× | 2.33× | 0.73× | 0.21× | 0.74 |
| rptest | t128 | Mops_cpu ↑ | 0.56 (0.49–0.588) | 0.291 (0.246–0.306) | 1.39 (1.2–1.66) | 1.79 (1.71–1.89) | 0.509 (0.409–0.648) | 0.056 (0.054–0.058) | 0.52× | 2.47× | 3.20× | 0.91× | 0.10× | 0.32 |

The comparators reproduce June to about ±10 %, the largest moves being mi
at t128 (1.79 vs 2.23, −20 %) and sys at t64 (0.56 vs 0.48, +17 %).  kame is 1.27× June at 1 thread, 1.75× at 4, 1.34× at 16,
1.46× at 64 and 1.75× at 128.  It now beats glibc (1.3–2.5×) and jemalloc
(1.1–1.4× from 4 threads; je leads only at 1) at every count, matches
tcmalloc at 4 and leads it from 16, while tcmalloc leads at 1 (1.22×).
mi / mi3 lead throughout, and by more as threads grow: 1.06–1.07× at 1,
1.14–1.40× at 4, 1.5–1.9× at 16, 1.8–2.3× at 64, 2.5–3.2× at 128.

**Peak RSS** (rptest's own report, MiB, 5 runs):

| threads | kame | sys | mi3 | mi | je | tc |
| ------: | ---: | --: | --: | -: | -: | -: |
| 16 | 209–240 | 51–53 | 98–137 | 134–167 | 171–189 | 69–90 |
| 64 | 941–1094 | 222–245 | 482–675 | 666–721 | 678–778 | 200–226 |
| 128 | 1980–2234 | 483–544 | 1226–1376 | 1190–1460 | 1307–1430 | 353–372 |

kame holds the most memory: about 1.5× mi / je at 64–128 threads and 4×
glibc — the price of the per-thread caches on a workload whose blocks
cross threads (the M5 note found the same, 186 vs 109 MiB at t10).

## 4. What changes in the README

- **bench_compare (three charts)**: nothing changes materially; the June
  figures stand within a few percent.  Optionally refresh to the c15u01n1
  numbers in §1 and fix the 4 MiB 1T bold (system leads).
- **alloc_tune_report**: the bucket tier is 10–45 % higher; the chunk tier at
  ≥ 16 threads is too noisy for one run to say much — report a range, and
  the 1 MiB / 128T figure (543 in June) is open.
- **xmalloc-test**: retitle the June table as 1024 B (that is what it ran),
  replace it with §3.1's 1024 B numbers, and add the 64 B table — where kame
  leads from 4 to 64 worker pairs and falls behind je at 128.
- **mstress / rptest**: replace; both improved, and kame now beats glibc and
  jemalloc throughout both.  Add rptest's peak-RSS cost.
- **larson** (new): 1–64 threads only — larson segfaults above 100 threads.


## 5. Re-measured at `e0974e402` (2026-10-13/14)

Every table again, on KAME master `55f125f66` — kamepoolalloc as of
`e0974e402`; the commits after it touch only READMEs, design notes, a
`contrib/` comment and `make_plots.py`.  Against §1–§3 (`19b2e4758`, the
v1.2.1 allocator) that adds the thread lists (`5fc34e052`), the per-thread
cap (`f9e5cb9b3` … `9f6370a52`) and the owner free's bucket read off `p`
(`78e57424c` for FS=true, `e0974e402` for the full tier) — the commits the M5
note's §4 and the Windows run in `THREAD_LIST_CHECK.md` measured.  Same
partition, kernel, compiler, comparator builds (sha256 as in §3.0) and
commands as §1–§3; the job script is `~/kame-claude/readme-e0974/job.sbatch`,
submitted one job at a time.

| job | SLURM | node | JST | what |
|---|---|---|---|---|
| bc128 | 3114748 | c15u04n1 | 10-13 23:20–23:24 | `bench_compare.sh --threads 128` |
| bc4 | 3114749 | c15u04n1 | 23:24–23:25 | `bench_compare.sh --threads 4` |
| tune | 3114750 | c15u01n1 | 23:25–23:29 | `alloc_tune_report` × 3 |
| xm64 | 3114751 | c15u01n2 | 23:29–23:41 | `bench_ab.sh --only xmalloc --xmalloc-sizes 64` |
| xm1024 | 3114752 | c15u01n2 | 23:41–23:54 | `bench_ab.sh --only xmalloc --xmalloc-sizes 1024` |
| larson | 3114753 | c15u01n2 | 23:55–10-14 00:09 | `bench_ab.sh --only larson --threads "1 4 16 64"` |
| rpt-1-4 | 3114757 | c15u02n1 | 00:09–00:25 | `bench_ab.sh --only rptest --threads "1 4"` |
| rpt-16-64 | 3114767 | c15u04n2 | 00:25–00:41 | `bench_ab.sh --only rptest --threads "16 64"` |
| mstress | 3114755 | c15u02n1 | 00:42–00:51 | `bench_ab.sh --only mstress` |
| rpt-128 | 3114776 | c15u01n1 | 00:51–01:21 | `bench_ab.sh --only rptest --threads 128` — **timed out**, see below |
| rpt-128 | 3114816 | c15u01n1 | 01:29–01:38 | the same, again |
| loopab | 3114835 | c15u05n2 | 01:43 | `bench_ab.sh --only loop`, tree against three earlier commits (§5.2) |
| tuneab | 3114836 | c15u01n4 | 01:46–01:54 | `alloc_tune_report`, tree against v1.2.1, interleaved (§5.3) |

Every other job exited 0 and every run reported `ok`; every job saw a clean
tree (`git status` empty under `kamepoolalloc/`).  The first rpt-128 hung in
the **mi** arm (mimalloc 1.8.2) of repetition 3: after 17 minutes the
`rptest 128 …` process had used 1 min 29 s of CPU over its 129 threads, all
of them asleep in `hrtimer_nanosleep` (`readme-e0974/mi_hang_probe.log`),
and `bench_ab.sh`'s per-run `timeout 1800` outlasts the job's 30 minutes, so
the job was cancelled at its limit with repetitions 1–2 complete (kame
0.545 / 0.572, mi 2.26 / 2.19 — in line with the rerun).  The rerun on the
same node completed all five repetitions, mi included.  The hang is
mimalloc's or rptest's, not kame's (kame's arm was not running); it has not
recurred.  The nodes differ from §1–§3's, which matters below where it does.

### 5.1 `bench_compare.sh`

Build tree `~/kame/build/tests` as in §1, rebuilt at `55f125f66`:
`libkamepoolalloc.so` 821304 B, sha256 `3debca37cf86eff5…`, `-O3 -DNDEBUG`
as before.  `alloc_tune_report` (sha256 `eb75bc31774ad8cc…`, unchanged)
loads that library through its RPATH.  bc128 and bc4 both landed on
`c15u04n1`, not §1's `c15u01n1` / `c15u02n1`.

#### 1T (median of 5, M ops/s) — bc128; §1 (`19b2e4758`, `c15u01n1`) in brackets

| size   | system    | mimalloc      | jemalloc  | kame              |
|--------|-----------|---------------|-----------|-------------------|
| 64B    | 212 (212) | **325** (327) | 181 (182) | 295 (259)         |
| 1.0KB  | 212 (210) | 209 (217)     | 159 (160) | **289** (228)     |
| 16KB   | 68 (69)   | 95 (95)       | 43 (45)   | **286** (173)     |
| 64KB   | 65 (65)   | **91** (95)   | 4 (4)     | 81 (79)           |
| 256KB  | 71 (71)   | **94** (95)   | 4 (4)     | 87 (80)           |
| 1.0MB  | 70 (71)   | 5 (5)         | 4 (4)     | **84** (83)       |
| 4.0MB  | **69** (70) | 4 (4)       | 4 (4)     | 58 (59)           |

bc4's 1T table, same node: 64 B 298, 1 KiB 285, 16 KiB 287, 64 KiB 81,
256 KiB 87, 1 MiB 84, 4 MiB 59 for kame; the comparators within 5 % of
bc128's (system 64 KiB 62 vs 65, the rest within 3 %).

#### 4 processes (aggregate, M ops/s, median of 5) — bc4; §1 (`c15u02n1`) in brackets

| size   | system    | mimalloc        | jemalloc  | kame            |
|--------|-----------|-----------------|-----------|-----------------|
| 64B    | 844 (825) | **1258** (1245) | 724 (711) | 1202 (1014)     |
| 16KB   | 271 (268) | 368 (371)       | 174 (172) | **1061** (681)  |
| 64KB   | 254 (253) | **374** (370)   | 16 (16)   | 320 (308)       |
| 1.0MB  | 275 (277) | 20 (20)         | 16 (16)   | **332** (326)   |

#### 128 processes (aggregate, M ops/s, median of 5) — bc128; §1 (`c15u01n1`) in brackets

| size   | system        | mimalloc          | jemalloc      | kame              |
|--------|---------------|-------------------|---------------|-------------------|
| 64B    | 22839 (22575) | **38244** (39396) | 21558 (21184) | 35432 (31006)     |
| 16KB   | 8105 (7995)   | 11662 (11772)     | 5314 (5263)   | **34847** (22001) |
| 64KB   | 8036 (8101)   | **11717** (11864) | 459 (469)     | 10278 (9903)      |
| 1.0MB  | 8813 (8917)   | 565 (562)         | 475 (473)     | **10627** (10567) |

The comparators reproduce §1 to within 4.4 % (mimalloc 64 KiB 1T −4 %, 1 KiB
1T −4 %, jemalloc 16 KiB 1T −4 %; everything else within 3 %), so
`c15u04n1` measures like `c15u01n1`, and on the 4-process table slightly
above `c15u02n1` (+1–2 %).  kame moved where the code did: single thread
64 B 1.14×, 1 KiB 1.27×, **16 KiB 1.65×**; 4 processes 1.19× / 1.56× at
64 B / 16 KiB; 128 processes 1.14× / 1.58×.  64 KiB – 4 MiB are level
(0.98–1.09×).  The orderings: kame now leads at 1 KiB (289 vs 209, 1.38×)
and 16 KiB (3.0× mimalloc single-thread and at 128 processes, 2.9× at 4) as
well as 1 MiB; mimalloc still leads at 64 B, by 1.10× single-thread (from
1.26×), 1.05× at 4 processes, 1.08× at 128; and at 64–256 KiB (91–94 vs
81–87); glibc at 4 MiB single-thread (69 vs 58).  §5.2 checks the 64 B / 1 KiB
/ 16 KiB moves interleaved.

### 5.2 `loop`, interleaved against v1.2.1 and the two commits before `e0974e402`

The Linux counterpart of the M5 note's §4.2, closing the "Linux has not
re-run `loop`" item in `THREAD_LIST_CHECK.md`.  The three earlier arms are
`git archive <commit> kamepoolalloc` built under
`~/kame-claude/bench-ab-arms/<commit>/` as `bench_ab.sh` builds a tag
(`cmake -DCMAKE_BUILD_TYPE=Release -DKAMEPOOLALLOC_BUILD_TESTS=OFF`, clang
18.1.8): `e2c9e6dde` (before `78e57424c`, sha256 `59d0ecf33d3b7da7`),
`6ae486851` (between `78e57424c` and `e0974e402`, `5b1d0f60166c3169`), v121 =
`45f1d07f9` (`e4603b3a2426fc38` — byte-identical to §3's `19b2e4758` arm,
as it should be).  Tree = the `kp-tree` arm of §5.4 (`58cc226d0a4bdd26`).

    bench_ab.sh --run-only --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench --tags tree \
        --arm e2c9e6dde=…/e2c9e6dde/kamepoolalloc/out/libkamepoolalloc.so \
        --arm 6ae486851=…/6ae486851/kamepoolalloc/out/libkamepoolalloc.so \
        --arm v121=…/45f1d07f9/kamepoolalloc/out/libkamepoolalloc.so \
        --arm mi3=$X/mi3/out/release/libmimalloc.so --only loop --reps 9

loopab, `c15u05n2`, `results-c15u05n2-20261014-014301`:

| bench | config | metric | kp-tree | e2c9e6dde | 6ae486851 | v121 | mi3 | e2c9e6dde / kp-tree | 6ae486851 / kp-tree | v121 / kp-tree | mi3 / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 305 (230–314) | 210 (141–225) | 303 (278–311) | 262 (261–316) | 327 (194–334) | 0.69× | 0.99× | 0.86× | 1.07× |
| loop | 1024B | Mops ↑ | 289 (196–290) | 289 (279–290) | 289 (196–289) | 229 (174–229) | 204 (180–211) | 1.00× | 1.00× | 0.79× | 0.71× |
| loop | 16384B | Mops ↑ | 287 (284–293) | 163 (144–165) | 166 (164–167) | 172 (170–174) | 94.9 (89.8–96.0) | 0.57× | 0.58× | 0.60× | 0.33× |

(A few repetitions in most arms ran slow — the minima; the medians of 9 sit
near the top of each range.)

- **`78e57424c` is 1.44× at 64 B** (6ae486851 against e2c9e6dde, 303 vs
  210), and **`e0974e402` 1.73× at 16 KiB** (287 vs 166); each leaves the
  other sizes level.  The same two steps as on the Windows Xeon (1.44×,
  2.04×), where the M5 Ultra showed neither — consistent with the dependent
  loads in front of the owner free's store costing most on cores without
  memory renaming (Zen 2, Broadwell-EP; the README's architecture note).
- **Against v1.2.1 the tree is 1.16× / 1.26× / 1.67×** at 64 B / 1 KiB /
  16 KiB.  The thread lists themselves (e2c9e6dde against v121) cost 0.80× at
  64 B and 0.95× at 16 KiB and gained 1.26× at 1 KiB; the two bucket-off-`p`
  commits turned both losses into gains.  These are §5.1's moves (1.14×,
  1.27×, 1.65×), so those are the code.
- mi3 / tree 1.07× at 64 B (§5.1: 1.10×), 0.71× at 1 KiB, 0.33× at 16 KiB.

### 5.3 `alloc_tune_report`

tune, `c15u01n1`: the report three times in one job (`TUNE_REPS=3`).
Aggregate M ops/s, median of the three runs (range); §2 (`19b2e4758`, one
run on `c15u04n4`, and tune3's three on `c15u01n2` where they exist) in
brackets:

| size,tier                  | 1T              | 4T                | 16T                         | 64T                         | 128T                          |
| -------------------------- | --------------: | ----------------: | --------------------------: | --------------------------: | ----------------------------: |
| 64 B  (bucket)             | 150 (145–151) [132] | 578 (575–590) [507] | 2370 (2358–2376) [2094–2097] | 8430 (8315–8779) [7457–7898] | 12441 (12323–13121) [11580–13924] |
| 1 KiB (bucket)             | 149 (149–149) [114] | 588 (586–589) [457] | 2364 (2362–2365) [1822–1826] | 8091 (7612–8229) [6379–6641] | 12301 (12285–12825) [10200–11429] |
| 64 KiB (chunk)             | 46 (46–46) [43] | 31 (29–31) [29–31] | 71 (71–72) [61–75]         | 212 (167–219) [171–204]     | 267 (257–355) [186–353]       |
| 1 MiB (chunk)              | 42 (42–42) [42] | **26 (26–27) [29–40]** | 214 (117–259) [127–218] | 238 (138–273) [135–237]     | 378 (338–380) [345–431]       |
| 8 MiB (large_va)           | 27 (27–27) [29] | 4.3 (4.0–7.6) [7.4] | 3.4 (3.3–3.7) [4.4]     | 2.9 (2.9–3.3) [3.6]         | 3.3 (2.8–4.3) [2.8]           |
| 40 MiB (large_va, cached)  | 12.5 (12.5–12.6) [12] | 4.3 (3.4–5.4) [2.6] | 3.9 (3.3–4.0) [3.7] | 2.6 (2.3–2.9) [2.2]       | 2.9 (2.6–6.2) [2.5]           |

(The 4T bracket for the chunk rows includes tune3's three runs, whose 4T
cells §2 did not tabulate: 64 KiB 28.6 / 30.9 / 29.7, 1 MiB 40.1 / 28.9 /
39.9.)

- The bucket tier is up by what §5.2 measured for a single thread — 64 B
  1.13–1.14× at 1–16T, 1 KiB 1.29–1.31× — a little less at 64T (64 B
  1.07–1.13×, 1 KiB 1.22–1.27× over §2's range), and at 128T 64 B is inside
  §2's range and 1 KiB 1.08–1.21× above it.  64 B / 128T is now 83× its 1-thread
  rate (§2: 94×), because the single thread got faster.
- The chunk tier from 16 threads is as noisy as in §2 and in the same
  ranges.
- **1 MiB at 4 threads: 26 in all three runs, against 29–40 (four runs,
  median about 38) at v1.2.1** — checked interleaved below: a loss of about
  0.8×, not yet profiled.  So is **8 MiB single-thread** (27 against §2's 29),
  which the interleaved run shows with disjoint ranges.
- Self-validation as before: `LRC_LAZY_INTERVAL_NS` 0.61 / 0.62 / 0.64 %
  ("default 10 ms is fine…"); `LRC_K_MAX` "128 cores: K_MAX=256 (default) is
  appropriate"; `MADV_HUGEPAGE` 0.24 µs/page as plain first-touch; TLB
  shootdown, 32 MiB `munmap` at 128 threads 2145 / 1812 / 1732 µs = 28.9× /
  26.0× / 10.6× the 1-thread cost (whose baseline was 74 / 70 / 163 µs).
  Over §2's four runs and these three the absolute cost is 1.4–2.1 ms.

#### Interleaved against v1.2.1 (tuneab)

`alloc_tune_report` built from `git archive` of `45f1d07f9` (v1.2.1) and of
`55f125f66` under `~/kame-claude/bench-ab-arms/{45f1d07f9,tree-55f125f66}/tune`
(CMake Release, `-DKAMEPOOLALLOC_BUILD_TESTS=ON`, clang 18.1.8, `-O3
-DNDEBUG`; each binary loads its own `libkamepoolalloc.so.8`), run four
times each in one job on `c15u01n4`, alternating which arm goes first
(`readme-e0974/tuneab.sbatch`).  M ops/s, median of 4 (range); cells not
listed overlap between the arms or are within 5 %:

| cell | v1.2.1 | tree | tree / v1.2.1 |
|---|---|---|---|
| 64 B, 1T | 131.3 (131.3–131.4) | 152.5 (151.7–153.1) | 1.16× |
| 1 KiB, 1T | 114.2 (113.9–114.3) | 148.9 (148.7–148.9) | 1.30× |
| 64 B, 4T / 16T / 64T | 507 / 2096 / 7753 | 586 / 2380 / 8574 | 1.16× / 1.14× / 1.11× |
| 1 KiB, 4T / 16T / 64T | 457 / 1826 / 6551 | 589 / 2363 / 8114 | 1.29× / 1.29× / 1.24× |
| 64 B, 128T | 11699 (11611–12022) | 12594 (12132–13035) | 1.08× |
| 1 KiB, 128T | 10920 (10359–11784) | 12337 (12054–12788) | 1.13× |
| 64 KiB, 1T | 43.3 (42.6–43.3) | 45.9 (45.9–46.0) | 1.06× |
| **1 MiB, 4T** | 34.7 (33.1–38.6) | 28.4 (26.2–40.3) | **0.82×** |
| **8 MiB, 1T** | 30.8 (28.2–31.0) | 26.8 (24.5–26.9) | **0.87×** |
| 40 MiB, 64T | 2.25 (2.05–2.37) | 2.82 (2.68–3.09) | 1.25× |

- The bucket tier's gain is the code's: the single-thread figures are
  §5.2's interleaved `loop` ratios (1.16×, 1.26–1.30×), and they carry to
  64 threads.
- **1 MiB at 4 threads** overlaps only through one tree run (40.3); with the
  tune job's three, 6 of 7 tree runs are at or below 29.3 and 7 of 8 v1.2.1
  runs (§2's four and these) at or above 33.  **8 MiB single-thread** is
  disjoint, 24.5–26.9 against 28.2–31.0.  Both are losses of the tree, about
  0.8× and 0.87×; neither is profiled nor bisected to a commit.  The chunk
  tier from
  16 threads stays too noisy to compare (1 MiB / 128T 192–359 against
  258–411).
- TLB shootdown at 128 threads, the eight runs: 1442–1770 µs.

### 5.4 mimalloc-bench suite (`bench_ab.sh`)

As §3.0, `--run-only` on the arms there; the kame arm rebuilt at
`55f125f66`: `libkamepoolalloc.so` 821304 B, sha256 `58cc226d0a4bdd26`.
Result directories under `~/kame-claude/bench-ab/`: xm64
`results-c15u01n2-20261013-232908`, xm1024 `…-c15u01n2-20261013-234159`,
larson `…-c15u01n2-20261013-235505`, rptest
`…-c15u02n1-20261014-000933` (t1, t4), `…-c15u04n2-20261014-002547` (t16,
t64), `…-c15u01n1-20261014-012959` (t128), mstress
`…-c15u02n1-20261014-004201`.  The timed-out first rpt-128 left
`…-c15u01n1-20261014-005125` with no `summary.md`.

**Nodes.**  xmalloc and larson ran on `c15u01n2`, which ran §3's larson,
mstress and rptest; xmalloc in §3 was on `c15u01n1`.  mstress and rptest ran
on other nodes than in §3, and there the comparators moved by up to a third
(rptest t16 je 3.25 vs 2.5, mstress t128 mi 5.04 vs 5.87 s); kame's changes
in those two benches are within what the node moves, and are not claimed.

#### xmalloc-test, 64 B (xm64; §3.1's kame in the last column)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §3.1 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,64B | Mfree ↑ | 32.9 (31.9–34.2) | 5.68 (5.65–5.69) | 34.1 (30.6–35.7) | 34.0 (34.0–34.8) | 29.3 (26.0–29.9) | 8.83 (6.17–8.96) | 0.17× | 1.04× | 1.03× | 0.89× | 0.27× | 31.9 |
| xmalloc | w4,64B | Mfree ↑ | 130 (128–131) | 15.8 (14.2–17.1) | 106 (100–113) | 97.2 (86.3–100) | 93.7 (89.6–98.1) | 1.61 (1.46–1.83) | 0.12× | 0.82× | 0.75× | 0.72× | 0.01× | 127 |
| xmalloc | w16,64B | Mfree ↑ | 503 (500–508) | 66.1 (54.2–67.6) | 325 (261–380) | 200 (184–283) | 335 (329–376) | 1.12 (1.04–1.17) | 0.13× | 0.65× | 0.40× | 0.67× | 0.00× | 494 |
| xmalloc | w64,64B | Mfree ↑ | 865 (720–1256) | 196 (186–206) | 416 (397–437) | 393 (360–443) | 875 (811–896) | 0.771 (0.758–0.791) | 0.23× | 0.48× | 0.45× | 1.01× | 0.00× | 1149 |
| xmalloc | w128,64B | Mfree ↑ | 368 (346–449) | 187 (185–190) | 440 (395–483) | 361 (323–424) | 881 (854–1003) | 0.844 (0.823–0.853) | 0.51× | 1.20× | 0.98× | 2.40× | 0.00× | 422 |

kame leads at 4 and 16 pairs (1.23× and 1.50× the next best), is level with
mi3 / mi at 1 and with je at 64 — where its range (720–1256) again spans
§3.1's 1149 — and at 128 falls to 368, behind je (881) and mi3 (440).  The
w64 spread and the w128 step remain kame's open items at 64 B.

#### xmalloc-test, 1024 B (xm1024)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §3.1 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,1024B | Mfree ↑ | 6.71 (5.84–6.76) | 3.26 (3.16–3.36) | 24.6 (22.9–26.1) | 22.9 (22.7–23.1) | 4.83 (4.76–5.05) | 3.02 (2.59–3.21) | 0.49× | 3.67× | 3.41× | 0.72× | 0.45× | 6.17 |
| xmalloc | w4,1024B | Mfree ↑ | 20.3 (19.5–21.8) | 9.33 (8.77–9.89) | 54.1 (49.4–57.2) | 63.7 (56.3–64.7) | 15.1 (14.0–15.9) | 1.14 (1.03–1.55) | 0.46× | 2.67× | 3.14× | 0.75× | 0.06× | 18.3 |
| xmalloc | w16,1024B | Mfree ↑ | 62.3 (55.9–82.0) | 24.4 (19.3–25.6) | 131 (119–143) | 193 (177–202) | 59.4 (55.7–63.8) | 0.952 (0.756–1) | 0.39× | 2.10× | 3.10× | 0.95× | 0.02× | 71.7 |
| xmalloc | w64,1024B | Mfree ↑ | 53.6 (47.2–73.2) | 12.3 (11.8–12.9) | 195 (193–204) | 258 (239–273) | 172 (163–189) | 0.619 (0.614–0.625) | 0.23× | 3.63× | 4.82× | 3.21× | 0.01× | 51.1 |
| xmalloc | w128,1024B | Mfree ↑ | 63.0 (55.3–76.2) | 8.36 (7.49–9.09) | 205 (192–206) | 172 (160–182) | 158 (116–165) | 0.606 (0.602–0.611) | 0.13× | 3.26× | 2.73× | 2.51× | 0.01× | 76.2 |

kame is within its run-to-run spread of §3.1 throughout (1.09–1.11× at 1–4
pairs; 16 and 128 pairs lower, each inside the other's range).  The leader
is mimalloc at every count, by 3.1–4.8× (the best of mi3 / mi over kame);
jemalloc passes kame from 64 pairs and is level at 16; kame beats glibc
2.1–7.5×.

#### larson (`c15u01n2`, as §3.2)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §3.2 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| larson | t1 | Mops ↑ | 28.7 (28.6–28.7) | 21.7 (20.3–21.8) | 33.1 (32.8–33.3) | 42.0 (42.0–43.0) | 36.2 (36.0–36.3) | 45.1 (44.6–45.2) | 0.76× | 1.15× | 1.47× | 1.26× | 1.57× | 22.9 |
| larson | t4 | Mops ↑ | 72.7 (65.9–79.2) | 49.7 (44.4–52.4) | 58.5 (55.0–59.0) | 99.9 (87.3–105) | 92.3 (82.3–104) | 94.9 (86.5–105) | 0.68× | 0.81× | 1.37× | 1.27× | 1.31× | 62.8 |
| larson | t16 | Mops ↑ | 263 (254–269) | 168 (163–171) | 231 (227–236) | 370 (359–376) | 347 (326–352) | 343 (324–349) | 0.64× | 0.88× | 1.41× | 1.32× | 1.30× | 224 |
| larson | t64 | Mops ↑ | 816 (802–831) | 588 (584–594) | 875 (873–883) | 1325 (1317–1347) | 1261 (1244–1269) | 1099 (1092–1107) | 0.72× | 1.07× | 1.62× | 1.54× | 1.35× | 650 |

Same node as §3.2, comparators within 2.5 % except mi3 t4 (58.5 vs 63.7) and
mi t4 (99.9 vs 105), and kame **1.16–1.26× faster**: 1.25× / 1.16× / 1.17× /
1.26× at 1 / 4 / 16 / 64 threads.  It is now ahead of mi3 at 4–16 threads
(1.24×, 1.14×; mi3 leads at 1 and 64, 1.15× and 1.07×) and of glibc by
1.32–1.56×, and behind mi / je / tc by 1.26–1.62×.  Not profiled.

#### mstress (`c15u02n1`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mstress | t1 | wall_s ↓ | 0.0578 (0.0565–0.0856) | 0.0699 (0.0682–0.0707) | 0.0553 (0.0549–0.0561) | 0.0543 (0.054–0.0546) | 0.06 (0.0596–0.0604) | 0.0575 (0.0568–0.0584) | 1.21× | 0.96× | 0.94× | 1.04× | 0.99× |
| mstress | t4 | wall_s ↓ | 0.543 (0.517–0.558) | 0.548 (0.542–0.556) | 0.439 (0.433–0.456) | 0.424 (0.421–0.43) | 0.807 (0.798–0.827) | 0.568 (0.56–0.588) | 1.01× | 0.81× | 0.78× | 1.49× | 1.05× |
| mstress | t16 | wall_s ↓ | 2.23 (2.22–2.26) | 2.48 (2.41–2.56) | 1.72 (1.63–1.84) | 1.83 (1.8–1.85) | 2.58 (2.55–2.58) | 2.04 (1.99–2.05) | 1.11× | 0.77× | 0.82× | 1.16× | 0.91× |
| mstress | t64 | wall_s ↓ | 6.02 (5.61–6.61) | 7.78 (7.59–7.99) | 3.77 (3.63–4.45) | 3.61 (3.43–3.85) | 8.04 (7.98–8.23) | 5.7 (5.52–6.04) | 1.29× | 0.63× | 0.60× | 1.33× | 0.95× |
| mstress | t128 | wall_s ↓ | 8.84 (8.26–9.06) | 14.1 (14.0–14.2) | 5.75 (5.56–6.19) | 5.04 (4.81–5.4) | 14.4 (14.3–14.6) | 9.85 (9.59–10.4) | 1.59× | 0.65× | 0.57× | 1.63× | 1.11× |

The ratios against §3.3 (`c15u01n2`) barely move: kame beats jemalloc
everywhere and glibc everywhere but 4 threads (level, 1.01×), is within
11 % of tcmalloc either way, and trails mi / mi3 by 1.3–1.75× from 4 threads
up (§3.3: 1.3–1.6×).

#### rptest (t1–t4 `c15u02n1`, t16–t64 `c15u04n2`, t128 `c15u01n1`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rptest | t1 | Mops_cpu ↑ | 14.5 (14.4–15.9) | 10.4 (10.3–11.0) | 15.2 (14.2–16.1) | 15.7 (15.2–16.2) | 15.4 (15.3–16.2) | 18.1 (17.3–18.3) | 0.71× | 1.05× | 1.08× | 1.06× | 1.24× |
| rptest | t4 | Mops_cpu ↑ | 7.59 (6.96–7.67) | 2.87 (2.62–2.97) | 7.14 (6.38–7.37) | 9.33 (8.83–10.3) | 5.94 (5.25–6.08) | 6.42 (6.12–6.62) | 0.38× | 0.94× | 1.23× | 0.78× | 0.85× |
| rptest | t16 | Mops_cpu ↑ | 3.7 (3.49–3.98) | 2.22 (2.04–2.28) | 4.82 (4.49–5.06) | 6.18 (5.45–6.87) | 3.25 (2.59–3.56) | 2.96 (2.47–3.23) | 0.60× | 1.30× | 1.67× | 0.88× | 0.80× |
| rptest | t64 | Mops_cpu ↑ | 1.2 (1.14–1.24) | 0.661 (0.635–0.774) | 2.17 (1.69–2.4) | 2.96 (2.64–3.02) | 0.961 (0.743–1.05) | 0.258 (0.251–0.341) | 0.55× | 1.82× | 2.47× | 0.80× | 0.22× |
| rptest | t128 | Mops_cpu ↑ | 0.578 (0.522–0.686) | 0.275 (0.252–0.346) | 1.44 (1.15–1.49) | 2.16 (2–2.27) | 0.672 (0.492–0.799) | 0.057 (0.055–0.06) | 0.48× | 2.50× | 3.73× | 1.16× | 0.10× |

kame beats glibc at every count (1.4–2.6×) and tcmalloc from 4 threads;
jemalloc at 4–64 threads (1.14–1.28×), while je leads at 1 (1.06×) and 128
(1.16×, within its 0.49–0.80 spread); kame passes mi3 at 4 threads (1.06×).
mimalloc leads, by more as threads grow: mi 1.08× kame at 1, 1.23× at 4,
1.67× at 16, 2.47× at 64, 3.73× at 128.  Against §3.4 (`c15u01n2`) kame's
1.0–1.2× is node-sized — mi moved from 1.79 to 2.16 at t128 and from 2.51 to
2.96 at t64 — so rptest is not where the thread lists show.

**Peak RSS** (rptest's own report, MiB, min–max of 5):

| threads | kame | sys | mi3 | mi | je | tc |
| ------: | ---: | --: | --: | -: | -: | -: |
| 16 | 225–259 | 49–53 | 99–117 | 129–177 | 163–184 | 72–97 |
| 64 | 961–1051 | 215–235 | 434–721 | 667–719 | 688–725 | 203–212 |
| 128 | 2034–2273 | 469–553 | 919–1517 | 1253–1729 | 1250–1356 | 348–371 |

kame's as §3.4 to within 8 %: the per-thread cap (4 MiB per bucket per
thread) did not change rptest's footprint — about 1.4–1.7× mi / je at 128
threads and 4× glibc.

### 5.5 What changes in the README

All Ohtaka tables, the three Ohtaka charts (`make_plots.py`, matplotlib
3.7.5 on the login node's Python 3.8 — the M5 chart stays as committed) and
their text, from §5.1–§5.4:

- **bench_compare**: kame leads 1 KiB as well now; 16 KiB 3.0× mimalloc;
  64 B gap 1.10× (from 1.26×), with §5.2's attribution.  The architecture
  note's 64 B pair becomes 295 vs 325.
- **alloc_tune_report**: medians of three, ranges for the chunk tier; 64 B /
  128T "83× its single-thread rate"; the two losses of §5.3 (1 MiB / 4T,
  8 MiB / 1T) stated.
- **xmalloc / larson / mstress / rptest**: replace; larson is the one that
  moved (1.16–1.26×, same node); rptest t128 now has jemalloc ahead of kame.

## 6. Re-measured at `deb2931e2` (2026-10-20)

Every table again, on KAME master `ba40008d5` — kamepoolalloc as of
`deb2931e2`.  Against §5 (`e0974e402`) the allocator adds §ded-fast
(`17a30aaae`, the dedicated tier, 32 KiB – 4 MiB, off the thread's page)
and its cross-thread follow-up (`deb2931e2`); the other allocator commits
in between are comments (`bead047a0`) and a gcc-only change (`c21a07f00`)
that this clang build does not see.  v1.3.0 (`433abfdf0`) carries the same
allocator as `e0974e402`: its `bench_ab.sh` arm is byte-identical to §5.4's
kame arm (sha256 `58cc226d0a4bdd26`), so §6.2's tree against v1.3.0 is
§ded-fast alone.  Same partition, kernel, compiler, comparator builds and
commands as §5; the job script is `~/kame-claude/readme-deb29/job.sbatch`
(`loopab.sbatch` for §6.2), the jobs submitted by `feeder.sh` as the
submit limit freed slots — they ran one after another, never two at once.

| job | SLURM | node | JST | what |
|---|---|---|---|---|
| bc128 | 3124314 | c15u04n4 | 10-20 19:55–20:05 | `bench_compare.sh --threads 128`, three runs |
| bc4 | 3124315 | c15u01n1 | 20:05–20:09 | `bench_compare.sh --threads 4`, three runs |
| loopab | 3124316 | c15u01n1 | 20:09–20:10 | `bench_ab.sh --only loop`, tree against v1.3.0 and v1.2.1 (§6.2) |
| tune | 3124317 | c15u01n3 | 20:10–20:13 | `alloc_tune_report` × 3 |
| xm64 | 3124318 | c15u01n1 | 20:13–20:26 | `bench_ab.sh --only xmalloc --xmalloc-sizes 64` |
| xm1024 | 3124323 | c15u03n3 | 20:26–20:39 | `bench_ab.sh --only xmalloc --xmalloc-sizes 1024` |
| larson | 3124324 | c15u01n1 | 20:39–20:53 | `bench_ab.sh --only larson --threads "1 4 16 64"` |
| mstress | 3124325 | c15u01n1 | 20:53–21:03 | `bench_ab.sh --only mstress` |
| rpt-1-4 | 3124336 | c15u01n4 | 21:03–21:19 | `bench_ab.sh --only rptest --threads "1 4"` |
| rpt-128 | 3124357 | c15u03n2 | 21:19–21:28 | `bench_ab.sh --only rptest --threads 128` |
| rpt-16-64 | 3124349 | c15u01n4 | 21:28–21:44 | `bench_ab.sh --only rptest --threads "16 64"` |

Every job exited 0, every run reported `ok`, and every job saw a clean tree
(`git status` empty under `kamepoolalloc/`).  rpt-128 completed all five
repetitions, mi included, in 8½ minutes: §5's mimalloc 1.8.2 hang did not
recur.  No cpufreq governor or boost control is exposed on these nodes;
`THP=always` throughout.

### 6.1 `bench_compare.sh`

Build tree `~/kame/build/tests` as in §5.1, rebuilt at `ba40008d5`:
`libkamepoolalloc.so` 821664 B, sha256 `3f64bb318398e539…`, `-O3 -DNDEBUG`.
Each job ran the script three times; **each cell is the median of the three
runs (each itself a median of 5)**, as the M5 and Windows tables now are.
Most cells agree to within 2 % over the three runs.  The exceptions:
kame 16 KiB single-thread in bc128's third run (220 against 286 / 286);
bc128's first run, which put the 128-process 64 B cells of system,
mimalloc and kame 3–7 % above the other two (kame 38592 against 36057 /
36118); and in bc4's second run
mimalloc 1 KiB single-thread (196 against 212 / 216) and kame 16 KiB on
4 processes (1090 against 1153 / 1152).

#### 1T (M ops/s) — bc128, `c15u04n4`; §5.1 (`e0974e402`, `c15u04n1`) in brackets

| size   | system    | mimalloc      | jemalloc  | kame              |
|--------|-----------|---------------|-----------|-------------------|
| 64B    | 209 (212) | **321** (325) | 180 (181) | 314 (295)         |
| 1.0KB  | 211 (212) | 209 (209)     | 158 (159) | **274** (289)     |
| 16KB   | 68 (68)   | 94 (95)       | 44 (43)   | **286** (286)     |
| 64KB   | 64 (65)   | 92 (91)       | 4 (4)     | **174** (81)      |
| 256KB  | 70 (71)   | 94 (94)       | 4 (4)     | **174** (87)      |
| 1.0MB  | 70 (70)   | 5 (5)         | 4 (4)     | **174** (84)      |
| 4.0MB  | **69** (69) | 4 (4)       | 4 (4)     | 60 (58)           |

bc4's 1T table, `c15u01n1`: kame 317 / 277 / 289 / 176 / 176 / 176 / 61;
the comparators within 2.2 % of bc128's but mimalloc 64 KiB (95 vs 92).

#### 4 processes (aggregate, M ops/s) — bc4, `c15u01n1`; §5.1 (`c15u04n1`) in brackets

| size   | system    | mimalloc        | jemalloc  | kame            |
|--------|-----------|-----------------|-----------|-----------------|
| 64B    | 850 (844) | **1320** (1258) | 728 (724) | 1264 (1202)     |
| 16KB   | 274 (271) | 370 (368)       | 176 (174) | **1152** (1061) |
| 64KB   | 258 (254) | 371 (374)       | 17 (16)   | **704** (320)   |
| 1.0MB  | 281 (275) | 20 (20)         | 17 (16)   | **703** (332)   |

#### 128 processes (aggregate, M ops/s) — bc128, `c15u04n4`; §5.1 (`c15u04n1`) in brackets

| size   | system        | mimalloc          | jemalloc      | kame              |
|--------|---------------|-------------------|---------------|-------------------|
| 64B    | 22443 (22839) | **38208** (38244) | 21411 (21558) | 36118 (35432)     |
| 16KB   | 7981 (8105)   | 11761 (11662)     | 5231 (5314)   | **35918** (34847) |
| 64KB   | 8053 (8036)   | 11801 (11717)     | 457 (459)     | **22152** (10278) |
| 1.0MB  | 8850 (8813)   | 563 (565)         | 467 (475)     | **22163** (10627) |

The comparators reproduce §5.1 to within 2.3 %, except mimalloc 64 B on
4 processes (+4.9 %, a different node) and jemalloc's 17 against 16 at
64 KiB / 1 MiB there (one unit).  kame moved where the code did:

- **The dedicated tier, 2.0–2.2×**: single thread 64 KiB / 256 KiB / 1 MiB
  2.15× / 2.00× / 2.07×; 4 processes 2.20× / 2.12× at 64 KiB / 1 MiB;
  128 processes 2.16× / 2.09×.  kame now leads mimalloc at 64–256 KiB too —
  174 vs 92–94 single-thread (1.9×), 704 vs 371 on 4 processes, 22152 vs
  11801 on 128 (1.88×) — and at 1 MiB / 128 processes by 39× (22163 vs
  563; jemalloc 47×).  4 MiB, above the tier, is level (1.03×); glibc
  still leads there single-thread (69 vs 60).
- **1 KiB single-thread 0.95×** (274 in all three runs against 289; bc4
  277), confirmed interleaved in §6.2 (0.96×).  The M5 Ultra (0.97×) and the
  Windows desktop (0.95×) showed the same cell; not profiled.
- 64 B 1.06× single-thread and 1.05× on 4 processes, 16 KiB 1.09× on
  4 processes; §6.2 has 64 B at 1.05× interleaved.  The rest is level.

The orderings: kame leads every size from 1 KiB to 1 MiB at every process
count; mimalloc leads only 64 B, by 1.02× single-thread (from 1.10×), 1.04×
on 4 processes and 1.06× on 128; glibc 4 MiB single-thread.

### 6.2 `loop`, interleaved against v1.3.0 and v1.2.1

loopab, `c15u01n1`, `results-c15u01n1-20261020-200941`: `bench_ab.sh
--run-only --tags v1.3.0,tree --arm v121=… --arm mi3=… --only loop --reps
9` (`loopab.sbatch`).  The tree arm is `kp-tree` of §6.4 (sha256
`7e381b562d72a04c`), v1.3.0 the `kp-v1.3.0` arm (`58cc226d0a4bdd26`, see
above), v121 §5.2's (`e4603b3a2426fc38`).

| bench | config | metric | kp-v1.3.0 | kp-tree | v121 | mi3 | kp-tree / kp-v1.3.0 | v121 / kp-v1.3.0 | mi3 / kp-v1.3.0 |
|---|---|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 301 (229–314) | 317 (265–327) | 262 (256–262) | 331 (321–334) | 1.05× | 0.87× | 1.10× |
| loop | 1024B | Mops ↑ | 289 (196–290) | 277 (266–278) | 229 (228–229) | 213 (189–220) | 0.96× | 0.79× | 0.74× |
| loop | 16384B | Mops ↑ | 286 (202–293) | 290 (222–290) | 174 (170–176) | 96.0 (80.8–96.2) | 1.01× | 0.61× | 0.33× |

- The v1.3.0 arm reproduces §5.2's tree (305 / 289 / 287) to 1.3 %, and
  v121 its own §5.2 row to 1.2 %: the node is comparable.
- **§ded-fast on the bucket sizes: 1.05× at 64 B, 0.96× at 1 KiB, level at
  16 KiB.**  At 1 KiB six of v1.3.0's nine repetitions ran 289 and none of
  the tree's passed 278 (266–278; v1.3.0's other three 196, 272, 277);
  §6.1's 0.95× is this.
- Against v1.2.1 the tree is 1.21× / 1.21× / 1.67× (§5.2: 1.16× / 1.26× /
  1.67×).  mi3 / tree 1.04× at 64 B, 0.77× at 1 KiB, 0.33× at 16 KiB.

### 6.3 `alloc_tune_report`

tune, `c15u01n3`, `TUNE_REPS=3` (`alloc_tune_report` sha256
`eb75bc31774ad8cc…`, unchanged, loading §6.1's library through its RPATH).
Aggregate M ops/s, median of the three runs (range); §5.3's median in
brackets:

| size,tier                  | 1T              | 4T                | 16T                  | 64T                  | 128T                    |
| -------------------------- | --------------: | ----------------: | -------------------: | -------------------: | ----------------------: |
| 64 B  (bucket)             | 147 (146–148) [150] | 572 (569–592) [578] | 2339 (2329–2379) [2370] | 8348 (8149–8686) [8430] | 12440 (12420–15317) [12441] |
| 1 KiB (bucket)             | 148 (147–148) [149] | 587 (568–588) [588] | 2363 (2327–2379) [2364] | 7894 (7746–8169) [8091] | 12327 (11016–12572) [12301] |
| 64 KiB (chunk)             | **86 (86–86) [46]** | **344 (334–345) [31]** | **1376 (1368–1386) [71]** | **4814 (4664–4880) [212]** | **7798 (7539–8029) [267]** |
| 1 MiB (chunk)              | **86 (86–86) [42]** | **346 (343–346) [26]** | **1388 (1378–1389) [214]** | **2008 (1993–2089) [238]** | **1514 (1452–1625) [378]** |
| 8 MiB (large_va)           | 27.5 (25.3–28.0) [27] | 4.7 (3.6–5.4) [4.3] | 3.4 (3.0–3.9) [3.4] | 3.5 (2.9–3.6) [2.9] | 3.9 (2.5–4.0) [3.3] |
| 40 MiB (large_va, cached)  | 12.4 (12.3–12.4) [12.5] | 3.5 (1.9–5.3) [4.3] | 2.9 (2.5–3.1) [3.9] | 2.1 (2.0–2.6) [2.6] | 2.8 (2.7–3.1) [2.9] |

- **The chunk tier is §ded-fast's**: 1.9–2.1× single-thread, and from
  4 threads it scales instead of collapsing — 64 KiB 344 → 1376 → 4814 →
  7798 (11×, 19×, 23×, 29× §5.3), 1 MiB 346 → 1388 at 4–16 threads (13×,
  6.5×).  The three runs now agree to within 7 % in every chunk cell but
  1 MiB / 128T (1452–1625, 12 %); §5.3 had to read those cells as ranges.
- **1 MiB stops scaling past 16 threads**: 2008 at 64 and 1514 at 128,
  while 64 KiB keeps climbing — not profiled.
- **§5.3's 1 MiB / 4T loss is gone** (346 against v1.2.1's ~35, about
  10×).  **8 MiB single-thread is not** (27.5 against v1.2.1's 30.8 in
  §5.3's tuneab): large_va is above the tier §ded-fast changed.
- The bucket rows are level with §5.3 (0.98–1.00× at every count; 64 B /
  128T's range takes one run at 15317).  `alloc_tune_report`'s loop does
  not show §6.2's 1 KiB 0.96× (148 vs 149).
- Self-validation as before: `LRC_LAZY_INTERVAL_NS` 0.71 / 0.68 / 0.69 %
  ("default 10 ms is fine…", all three); `LRC_K_MAX` "128 cores: K_MAX=256
  (default) is appropriate"; `MADV_HUGEPAGE` 0.33–0.35 µs/page, as plain
  first-touch in the same run; TLB shootdown, 32 MiB `munmap` at 128
  threads 1716 / 1521 / 2072 µs = 27.2× / 22.3× / 31.8× the 1-thread cost
  (63 / 68 / 65 µs).

### 6.4 mimalloc-bench suite (`bench_ab.sh`)

As §5.4, `--run-only` on the same comparator arms; the kame arm rebuilt at
`ba40008d5`: `libkamepoolalloc.so` 821664 B, sha256 `7e381b562d72a04c`.
Result directories under `~/kame-claude/bench-ab/`: xm64
`results-c15u01n1-20261020-201333`, xm1024 `…-c15u03n3-20261020-202623`,
larson `…-c15u01n1-20261020-203927`, mstress `…-c15u01n1-20261020-205357`,
rptest `…-c15u01n4-20261020-210326` (t1, t4), `…-c15u01n4-20261020-212826`
(t16, t64), `…-c15u03n2-20261020-211944` (t128).

**Summary: kame is level with §5.4 in every bench** — within its own
run-to-run spread or within what the comparators moved on the other nodes —
as it should be for workloads whose blocks are almost all below the
dedicated tier's 32 KiB.  The rows are below for the README; nothing in
this section is claimed as a change.

#### xmalloc-test, 64 B (xm64, `c15u01n1`; §5.4's kame in the last column)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §5.4 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,64B | Mfree ↑ | 32.9 (32.5–35.1) | 4.91 (4.32–5.72) | 34.7 (33.8–35.1) | 34.6 (34.0–34.8) | 28.7 (27.9–29.7) | 8.52 (6.75–9.01) | 0.15× | 1.06× | 1.05× | 0.87× | 0.26× | 32.9 |
| xmalloc | w4,64B | Mfree ↑ | 131 (129–133) | 15.8 (14.7–16.6) | 108 (105–109) | 99.9 (94.5–104) | 91.8 (88.9–100) | 1.57 (1.52–1.86) | 0.12× | 0.82× | 0.76× | 0.70× | 0.01× | 130 |
| xmalloc | w16,64B | Mfree ↑ | 509 (506–516) | 60.9 (53.6–65.1) | 316 (215–371) | 196 (169–336) | 367 (326–396) | 1.13 (1.11–1.16) | 0.12× | 0.62× | 0.38× | 0.72× | 0.00× | 503 |
| xmalloc | w64,64B | Mfree ↑ | 790 (736–961) | 195 (168–197) | 423 (277–452) | 392 (342–425) | 884 (858–906) | 0.778 (0.766–0.79) | 0.25× | 0.54× | 0.50× | 1.12× | 0.00× | 865 |
| xmalloc | w128,64B | Mfree ↑ | 411 (376–542) | 184 (183–187) | 417 (380–467) | 360 (289–411) | 901 (805–936) | 0.864 (0.823–0.864) | 0.45× | 1.02× | 0.87× | 2.19× | 0.00× | 368 |

kame leads at 4 and 16 pairs (1.21× mi3, 1.39× je), is within 6 % of mi3 /
mi at 1 pair, and behind je at 64 (1.12×) and 128 (2.19×, mi3 level there).
w64 and w128 are inside §5.4's ranges (720–1256, 346–449); the w64 spread
and the w128 step remain the open items.

#### xmalloc-test, 1024 B (xm1024, `c15u03n3`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §5.4 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w1,1024B | Mfree ↑ | 6.48 (5.92–6.95) | 3.03 (2.96–3.52) | 23.9 (23.1–26.2) | 22.6 (19.7–23.0) | 5.07 (4.76–5.12) | 3.14 (2.67–3.22) | 0.47× | 3.69× | 3.48× | 0.78× | 0.48× | 6.71 |
| xmalloc | w4,1024B | Mfree ↑ | 20.2 (19.7–20.6) | 9.35 (8.26–9.5) | 55.7 (51.2–59.0) | 61.8 (61.5–62.5) | 15.5 (13.2–16.0) | 1.15 (1.13–1.16) | 0.46× | 2.75× | 3.05× | 0.77× | 0.06× | 20.3 |
| xmalloc | w16,1024B | Mfree ↑ | 67.2 (56.3–79.2) | 18.9 (17.5–26.7) | 140 (137–144) | 188 (185–203) | 61.4 (53.7–63.7) | 0.969 (0.833–0.987) | 0.28× | 2.08× | 2.80× | 0.91× | 0.01× | 62.3 |
| xmalloc | w64,1024B | Mfree ↑ | 51.2 (43.2–72.3) | 12.7 (11.6–14.8) | 205 (184–209) | 256 (238–266) | 165 (146–181) | 0.622 (0.613–0.639) | 0.25× | 4.01× | 5.00× | 3.22× | 0.01× | 53.6 |
| xmalloc | w128,1024B | Mfree ↑ | 65.6 (63.6–86.8) | 8.11 (7.32–9.34) | 204 (195–207) | 169 (151–176) | 160 (153–161) | 0.602 (0.597–0.605) | 0.12× | 3.11× | 2.58× | 2.44× | 0.01× | 63.0 |

mimalloc leads at every count, by 2.8–5.0× (the better of mi3 / mi over
kame); jemalloc passes kame from 64 pairs (3.2×, 2.4×) and trails it at 16
(0.91×); kame beats glibc 2.1–8.1×.

#### larson (`c15u01n1`; §5.4 ran on `c15u01n2`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree | §5.4 kame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| larson | t1 | Mops ↑ | 28.6 (28.4–28.8) | 21.7 (21.5–21.8) | 32.9 (32.6–33.0) | 41.4 (41.1–41.5) | 36.1 (35.9–36.4) | 44.9 (44.8–45.3) | 0.76× | 1.15× | 1.45× | 1.26× | 1.57× | 28.7 |
| larson | t4 | Mops ↑ | 74.8 (65.0–83.8) | 50.9 (44.8–58.8) | 64.8 (63.2–76.1) | 113 (87.9–115) | 95.6 (78.6–102) | 99.3 (88.4–119) | 0.68× | 0.87× | 1.51× | 1.28× | 1.33× | 72.7 |
| larson | t16 | Mops ↑ | 269 (259–273) | 170 (168–172) | 229 (227–234) | 385 (366–401) | 350 (337–360) | 339 (334–350) | 0.63× | 0.85× | 1.43× | 1.30× | 1.26× | 263 |
| larson | t64 | Mops ↑ | 818 (812–823) | 592 (588–598) | 886 (878–892) | 1337 (1329–1351) | 1255 (1246–1269) | 1104 (1094–1108) | 0.72× | 1.08× | 1.63× | 1.53× | 1.35× | 816 |

kame ahead of glibc (1.32–1.58×) and of mi3 at 4–16 threads (1.15×,
1.17×); behind mi / je / tc by 1.26–1.63×.  The comparators are within 5 %
of §5.4 at 1, 16 and 64 threads; at 4, the noisiest row (kame 65.0–83.8),
mi3 / mi came out 11 % / 13 % higher.

#### mstress (`c15u01n1`; §5.4 `c15u02n1`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mstress | t1 | wall_s ↓ | 0.0574 (0.057–0.0809) | 0.0701 (0.0688–0.0736) | 0.0543 (0.0535–0.0549) | 0.0542 (0.0539–0.0543) | 0.0596 (0.0591–0.0603) | 0.0574 (0.057–0.081) | 1.22× | 0.95× | 0.94× | 1.04× | 1.00× |
| mstress | t4 | wall_s ↓ | 0.55 (0.525–0.566) | 0.567 (0.534–0.626) | 0.435 (0.43–0.453) | 0.423 (0.421–0.46) | 0.84 (0.805–0.926) | 0.607 (0.585–0.628) | 1.03× | 0.79× | 0.77× | 1.53× | 1.10× |
| mstress | t16 | wall_s ↓ | 2.24 (2.21–2.32) | 2.41 (2.37–2.52) | 1.83 (1.69–1.84) | 1.85 (1.73–1.92) | 2.6 (2.52–2.71) | 2.11 (2.04–2.2) | 1.07× | 0.82× | 0.82× | 1.16× | 0.94× |
| mstress | t64 | wall_s ↓ | 6.12 (5.74–6.46) | 7.74 (7.6–7.85) | 3.82 (3.59–4.05) | 3.67 (3.56–3.7) | 8.29 (8.14–8.44) | 5.59 (5.44–6.23) | 1.26× | 0.62× | 0.60× | 1.36× | 0.91× |
| mstress | t128 | wall_s ↓ | 9.19 (8.18–10.6) | 13.9 (13.5–14.3) | 5.66 (5.18–6.03) | 5.11 (4.99–5.4) | 14.5 (14.3–15.1) | 10.4 (9.79–10.5) | 1.51× | 0.62× | 0.56× | 1.57× | 1.13× |

As §5.4: kame beats jemalloc everywhere and glibc everywhere (4 threads
1.03×, about level), is within 13 % of tcmalloc either way, and trails
mi / mi3 by 1.2–1.8× from 4 threads up.

#### rptest (t1–t64 `c15u01n4`, t128 `c15u03n2`)

| bench | config | metric | kp-tree | sys | mi3 | mi | je | tc | sys / kp-tree | mi3 / kp-tree | mi / kp-tree | je / kp-tree | tc / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rptest | t1 | Mops_cpu ↑ | 15.0 (14.3–15.6) | 10.8 (10.6–11.3) | 15.2 (14.6–15.8) | 15.8 (15.1–15.9) | 16.4 (15.7–17.0) | 17.9 (16.7–19.1) | 0.72× | 1.01× | 1.05× | 1.09× | 1.19× |
| rptest | t4 | Mops_cpu ↑ | 7.23 (6.96–7.62) | 2.9 (2.84–3.27) | 7.87 (7.13–8.34) | 9.71 (9.29–10.6) | 6.17 (5.64–6.64) | 6.4 (5.6–6.67) | 0.40× | 1.09× | 1.34× | 0.85× | 0.88× |
| rptest | t16 | Mops_cpu ↑ | 3.5 (3.46–3.88) | 2.29 (1.97–2.37) | 4.62 (4.39–4.65) | 5.88 (5.08–6.42) | 2.6 (2.48–2.99) | 2.33 (2.01–2.8) | 0.66× | 1.32× | 1.68× | 0.74× | 0.67× |
| rptest | t64 | Mops_cpu ↑ | 1.1 (1.03–1.21) | 0.597 (0.572–0.691) | 1.96 (1.64–2.06) | 2.78 (2.46–2.92) | 0.815 (0.725–0.911) | 0.237 (0.191–0.266) | 0.54× | 1.78× | 2.52× | 0.74× | 0.21× |
| rptest | t128 | Mops_cpu ↑ | 0.556 (0.417–0.642) | 0.326 (0.314–0.438) | 1.33 (1.25–1.46) | 2.27 (1.65–2.54) | 0.62 (0.56–0.752) | 0.054 (0.054–0.057) | 0.59× | 2.40× | 4.08× | 1.12× | 0.10× |

kame 1.03× / 0.95× / 0.95× / 0.92× / 0.96× §5.4's at 1 / 4 / 16 / 64 /
128 threads, with mi 1.01× / 1.04× / 0.95× / 0.94× / 1.05× on the other
nodes: node-sized from 16 threads, and at 4 kame's range overlaps §5.4's
(6.96–7.62 against 6.96–7.67).  kame beats glibc at every count (1.4–2.5×), tcmalloc
from 4 threads, jemalloc at 4–64 (1.18–1.35×); je leads at 1 (1.09×) and
128 (1.12×, inside its 0.56–0.75 spread), tc at 1 (1.19×).  mi3 now leads
at 4 threads too (1.09×; §5.4 0.94×, inside both ranges).  mimalloc leads
by more as threads grow: mi 1.05× kame at 1, 1.34× at 4, 1.68× at 16,
2.52× at 64, 4.08× at 128.

**Peak RSS** (rptest's own report, MiB, min–max of 5):

| threads | kame | sys | mi3 | mi | je | tc |
| ------: | ---: | --: | --: | -: | -: | -: |
| 16 | 217–265 | 51–60 | 100–123 | 130–162 | 177–202 | 68–89 |
| 64 | 973–1138 | 217–230 | 490–590 | 649–708 | 721–766 | 191–210 |
| 128 | 1991–2173 | 427–522 | 1266–1508 | 1127–1365 | 1260–1331 | 349–383 |

kame's as §5.4 to within 9 %; 1.4–1.7× mi / je at 64–128 threads and
4–5× glibc.

### 6.5 What changes in the README

All Ohtaka tables, their text, and the three Ohtaka charts
(`make_plots.py`; matplotlib 3.7.5, as the committed SVGs, in a scratch venv
on the login node's Python 3.8):

- **bench_compare**: medians of three runs; kame leads 64 KiB – 1 MiB at
  every process count (1.9× mimalloc at 64–256 KiB, 39× at 1 MiB / 128
  processes); the 64 B gap is 1.02× single-thread; the 1 KiB 0.95× is
  stated.  The architecture note's 64 B pair becomes 314 vs 321.
- **loop**: tree against v1.3.0 (§ded-fast alone) and v1.2.1.
- **alloc_tune_report**: the chunk rows replace the ranges with medians;
  the 1 MiB / 4T loss is struck, the 8 MiB / 1T one kept, the 1 MiB
  plateau past 16 threads added.
- **xmalloc / larson / mstress / rptest**: replace; nothing moved.
- §ded-fast's closing sentence: the Ohtaka tables are re-measured.
