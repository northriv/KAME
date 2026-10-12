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
on 2026-10-12.

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

