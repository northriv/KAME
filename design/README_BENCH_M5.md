# The README's benchmarks on the Mac Studio M5 Ultra (2026-10-11)

Re-measurement of the tables in `kamepoolalloc/README.md` "Benchmarks"
(Apple MacBook Air M3) on the Mac Studio M5 Ultra, to replace them.  KAME
master `c9fa20d33` (kamepoolalloc as of `2e8f940fd`, the v1.2.0 regression
fix; the commits since change only tests and design notes).  §4 measures all
three sets again at `e0974e402` (2026-10-13), with the thread lists and the
per-thread cap; the README quotes §4.

**Comparators** — the versions the M3 tables name (mimalloc 3.3.2, jemalloc
5.3.0, "via Homebrew").  Homebrew now ships 3.5.4 / 5.4.0, so both were built
from their release sources with Homebrew's recipe, kept outside the system
(`~/bench-ab/alloc`), and passed explicitly:

- mimalloc 3.3.2 (`MI_MALLOC_VERSION 30302`): `v3.3.2.tar.gz` from
  github.com/microsoft/mimalloc (sha256 `ca02384e…bbbb`), `cmake
  -DCMAKE_BUILD_TYPE=Release -DMI_INSTALL_TOPLEVEL=ON`, `libmimalloc.dylib`.
- jemalloc 5.3.0 (`5.3.0-0-g54eaed1d8b56`): `jemalloc-5.3.0.tar.bz2` from the
  GitHub release (sha256 `2db82d1e…eaa`, the official one), `./configure
  --disable-debug --with-jemalloc-prefix=` && `make`,
  `libjemalloc.2.dylib`.  Active when preloaded (`MALLOC_CONF=stats_print:true`
  prints its statistics); on macOS it serves `malloc` through a malloc zone,
  hence its closeness to the system column, as on the M3.
- system: macOS 27.0.1 (26A434) `libsystem_malloc` (886.0.8).

**Machine** — Mac17,15, Apple M5 Ultra (10P + 20E), 96 GiB, macOS 27.0.1
(26A434), AC power; Apple clang 21.0.0 (clang-2100.3.34.2), CMake 3.31.12,
Release.  Not idle: Chrome Remote Desktop's host used about two cores
throughout (load average 3.4–5.6), as in `V1_2_PERF_HANDOFF` §4.1.

## 1. `bench_compare.sh` (the README's 1-thread and 4-process tables)

    cd kamepoolalloc/tests/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
    bash ../bench/bench_compare.sh --runs 7 \
        --mimalloc ~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --jemalloc ~/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib

(bash 5.3 from MacPorts; the script needs bash 4+.)  2026-10-11 10:30–10:31 UTC.

kamepoolalloc bench_compare: HEAD=c9fa20d33  host=dhcp-029-018.issp.u-tokyo.ac.jp/arm64  runs=7  os=Darwin

### 1T (median of 7, M ops/s)

| size      |  system | mimalloc | jemalloc |     kame |
|-----------|---------|----------|----------|----------|
| 64B       |     162 |     1055 |      168 |     1135 |
| 1 KiB     |     135 |      766 |      164 |      804 |
| 16 KiB    |     142 |      356 |      105 |      891 |
| 64 KiB    |      28 |      354 |       29 |      192 |
| 256 KiB   |      28 |      360 |       29 |      180 |
| 1 MiB     |      28 |        8 |       29 |      167 |
| 4 MiB     |      46 |        8 |       28 |      149 |

### 4 processes (aggregate, M ops/s, median of 7)

| size      |  system | mimalloc | jemalloc |     kame |
|-----------|---------|----------|----------|----------|
| 64B       |     641 |     3981 |      619 |     4258 |
| 16 KiB    |     525 |     1341 |      386 |     3340 |
| 64 KiB    |     106 |     1337 |      109 |      721 |
| 1 MiB     |     105 |       23 |      109 |      645 |

## 2. mimalloc-bench (`bench_ab.sh`)

    kamepoolalloc/tests/bench/bench_ab.sh --tags tree --arm sys= \
        --arm mi=~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --arm je=~/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib \
        --only "xmalloc larson rptest" --threads 10 --reps 5 \
        --work ~/bench-ab --mbench ~/bench-ab/mimalloc-bench

10 threads = the P cores (xmalloc runs 10 producers + 10 consumers).  Every
run produced a result.  `kp-tree` is the checkout above built with
mimalloc-bench's recipe (its sha256 matches the `fix` arm of
`V1_2_PERF_HANDOFF` §5).

| bench | config | metric | kp-tree | sys | mi | je | sys / kp-tree | mi / kp-tree | je / kp-tree |
|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w10 | Mfree ↑ | 489 (486–494) | 249 (246–272) | 501 (494–505) | 400 (400–401) | 0.51× | 1.02× | 0.82× |
| larson | t10 | Mops ↑ | 267 (265–272) | 26.1 (25.7–26.5) | 222 (216–230) | 425 (419–443) | 0.10× | 0.83× | 1.59× |
| rptest | t10 | Mops_cpu ↑ | 2.4 (2.24–2.56) | 2.69 (2.49–2.8) | 5.36 (4.76–5.61) | 5.78 (5.19–5.95) | 1.12× | 2.23× | 2.41× |

`provenance.txt`:

    date      2026-10-11T10:31:28Z
    host      dhcp-029-018.issp.u-tokyo.ac.jp
    kernel    Darwin 27.0.0 arm64
    cpu       Apple M5 Ultra, 10P + 20E, 96 GiB, Mac17,15
    os        macOS 27.0.1 (26A434)
    power     Now drawing from 'AC Power'
    benches   /Users/ssp/KAME/kamepoolalloc @ c9fa20d33, /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ (Apple clang version 21.0.0 (clang-2100.3.34.2))
    arm       kp-tree: /Users/ssp/bench-ab/kp-tree/out/libkamepoolalloc.dylib (685208 B, sha256 6f14cba89be3459c, tree /Users/ssp/KAME/kamepoolalloc at c9fa20d33, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    arm       sys: no preload (system malloc)
    arm       mi: /Users/ssp/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib (233072 B, sha256 5153705f94d7d8b4)
    arm       je: /Users/ssp/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib (649248 B, sha256 9290fc4e3540a8f2)
    mbench    /Users/ssp/bench-ab/mimalloc-bench @ 69c41ed, binaries in /Users/ssp/bench-ab/mbench-build
    params    reps=5 threads="10" only="xmalloc larson rptest" quick=0 xlat="1024x200000 1024x1000000 3000x200000" xlat_cpu=none loop_iters=20000000

### 2.1 rptest with `calloc` / `posix_memalign` interposed (2026-10-12)

The rptest row above did not measure kame.  rptest allocates only through
`calloc` and `posix_memalign` (its `benchmark_malloc`, `__MACH__` branch), and
the macOS dylib interposed `malloc` / `free` / `realloc` / `malloc_size` only:
a `sample` of the kp-tree run shows every allocation in libsystem's xzone
allocator and kame only in `free` (`kame_free` → `libsystem_free_for_pool`).
The row was system malloc plus the `free` forwarding, hence just under `sys`.
xmalloc-test, larson and mstress use `malloc` only and are unaffected; on
Linux the strong symbols always covered `calloc` / `memalign`.

With `calloc`, `posix_memalign` and `aligned_alloc` added to the dylib's
interpose table (same command, rptest only, its own work dir):

    kamepoolalloc/tests/bench/bench_ab.sh --tags tree --arm sys= \
        --arm mi=~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --arm je=~/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.dylib \
        --only "rptest" --threads 10 --reps 5 \
        --work ~/bench-ab-rpt --mbench ~/bench-ab/mimalloc-bench

| bench | config | metric | kp-tree | sys | mi | je | sys / kp-tree | mi / kp-tree | je / kp-tree |
|---|---|---|---|---|---|---|---|---|---|
| rptest | t10 | Mops_cpu ↑ | 6.29 (5.55–7.17) | 2.64 (2.48–2.83) | 5.77 (4.86–6.21) | 6.2 (5.18–6.42) | 0.42× | 0.92× | 0.99× |

An interleaved A/B of the old and new dylibs beside the others (t1 and t10,
median of 5): old 8.69 / 2.27, new 11.1 / 5.69, sys 9.3 / 2.49, mi 11.1 /
5.37, je 9.29 / 5.21.  Peak RSS at t10 (one run, rptest's own report): kame
186 MiB, mi 109, je 93 — kame holds more.

`provenance.txt`:

    date      2026-10-11T15:26:06Z
    host      dhcp-029-018.issp.u-tokyo.ac.jp
    kernel    Darwin 27.0.0 arm64
    cpu       Apple M5 Ultra, 10P + 20E, 96 GiB, Mac17,15
    os        macOS 27.0.1 (26A434)
    power     Now drawing from 'AC Power'
    benches   /Users/ssp/kame-rpt/kamepoolalloc @ 327644452, Apple clang version 21.0.0 (clang-2100.3.34.2)
    arm       kp-tree: libkamepoolalloc.dylib (685528 B, sha256 15903cd5c08a8c1f, tree at 327644452-dirty = this change)
    arm       sys: no preload (system malloc)
    arm       mi: libmimalloc.dylib (233072 B, sha256 5153705f94d7d8b4)
    arm       je: libjemalloc.dylib (649248 B, sha256 9290fc4e3540a8f2)
    mbench    mimalloc-bench @ 69c41ed
    params    reps=5 threads="10" only="rptest"

## 3. RT (`bench_rt_wcet`)

    kamepoolalloc/tests/build/bench_rt_wcet --pressure
    kamepoolalloc/tests/build/bench_rt_wcet --xt 2000000

`--xt 2000000` is the cross-thread sample count `--full` uses (the default is
120000), so the p99.99 the README quotes rests on 2 M samples.  The README's
rows, in its layout (3 interferers, Release, M5 Ultra):

| Path | realtime | default |
|---|---:|---:|
| free, > 256 MiB band — median | 20,480 ns | 20,480 ns |
| free, > 256 MiB band — max | 43,750 ns | 52,458 ns |
| free, 32 B cross-thread — p99.9 | **192 ns** | 224 ns |
| free, 32 B cross-thread — p99.99 | **512 ns** | 7,168 ns |
| free, 32 B cross-thread — p99.999 | **1,792 ns** | 10,240 ns |

(> 256 MiB rows from `--pressure`, n = 800 per arm; cross-thread rows from
`--xt 2000000`.  The `--pressure` run's own 120 K cross-thread samples:
p99.9 128 / 192 ns, p99.99 192 / 10,240 ns.)

**The > 256 MiB rows no longer show the realtime gate.**  The realtime arm
deferred 12 of its 800 frees (`deferred_unmaps=12`); the M3 row deferred all
of them (`RT_READINESS.md` G7 Result 1: `deferred_unmaps = 900`, free median
128 ns).  The difference is the deferral backlog cap (G5, `0834ee2bf`,
default 1 GiB): three 300 MiB blocks fit under it, the harness drains once per
round, and it runs 4 rounds — 4 × 3 = 12.  The other frees were forced inline
releases, the cap's documented trade.  Not a v1.2 change: `34799c70d`
(v1.2.0's code) gives the same 12 and the same 14–20 µs medians.  To show the
gate again the harness would have to drain within a round or raise the cap
(`kame_pool_set_rt_pending_cap`) for this band — done in §3.1.

**`--xt 2000000` ended `FAILED`: `rt_violations=2`** — the realtime arm made
two new mappings despite prewarming.  The measurement itself completed; the
same 2 violations appear on every build with the default / `--full`
arguments, master included, and were noted before this run (ctest's smaller
arguments pass).  §3.1 found their cause: a broken guarantee, not noise.

`--pressure` output:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke +pressure reps=4 iters=200 interferers=3
    pressure: added the > LRC_HI band, where the recycle cache is bypassed and frees really munmap
    clock overhead: mean=18 max=166 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=18      p50=0 p99=42 p99.9=42  MAX=2209 ns
        OFF malloc             n=51200      mean=18      p50=0 p99=42 p99.9=42  MAX=9167 ns
        RT  1st-touch          n=51200      mean=17      p50=0 p99=42 p99.9=42  MAX=3583 ns
        OFF 1st-touch          n=51200      mean=17      p50=0 p99=42 p99.9=42  MAX=6292 ns
        RT  free               n=51200      mean=22      p50=0 p99=192 p99.9=384  MAX=4083 ns
        OFF free               n=51200      mean=19      p50=0 p99=42 p99.9=128  MAX=12000 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=19      p50=0 p99=42 p99.9=96  MAX=708 ns
        OFF malloc             n=25600      mean=18      p50=0 p99=42 p99.9=42  MAX=375 ns
        RT  1st-touch          n=25600      mean=18      p50=0 p99=42 p99.9=42  MAX=83 ns
        OFF 1st-touch          n=25600      mean=17      p50=0 p99=42 p99.9=42  MAX=84 ns
        RT  free               n=25600      mean=20      p50=0 p99=42 p99.9=96  MAX=8084 ns
        OFF free               n=25600      mean=18      p50=0 p99=42 p99.9=96  MAX=84 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=51      p50=41 p99=384  MAX=833 ns
        OFF malloc             n=6400       mean=28      p50=41 p99=42  MAX=1416 ns
        RT  1st-touch          n=6400       mean=19      p50=0 p99=42  MAX=1375 ns
        OFF 1st-touch          n=6400       mean=19      p50=0 p99=42  MAX=1167 ns
        RT  free               n=6400       mean=25      p50=41 p99=128  MAX=209 ns
        OFF free               n=6400       mean=20      p50=41 p99=42  MAX=125 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=143     p50=128 p99=512  MAX=10000 ns
        OFF malloc             n=1600       mean=131     p50=128 p99=256  MAX=8292 ns
        RT  1st-touch          n=1600       mean=19      p50=0 p99=42  MAX=1083 ns
        OFF 1st-touch          n=1600       mean=18      p50=0 p99=42  MAX=916 ns
        RT  free               n=1600       mean=138     p50=128 p99=512  MAX=750 ns
        OFF free               n=1600       mean=126     p50=128 p99=320  MAX=459 ns

      300 MiB (> LRC_HI)
        RT  malloc             n=800        mean=3579    p50=3072  MAX=22833 ns
        OFF malloc             n=800        mean=3740    p50=3584  MAX=26834 ns
        RT  1st-touch          n=800        mean=786     p50=640  MAX=15125 ns
        OFF 1st-touch          n=800        mean=715     p50=768  MAX=5291 ns
        RT  free               n=800        mean=19302   p50=20480  MAX=43750 ns
        OFF free               n=800        mean=19521   p50=20480  MAX=52458 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=120000     mean=26      p50=41 p99=96 p99.9=128 p99.99=192  MAX=7458 ns
        OFF free               n=120000     mean=22      p50=0 p99=42 p99.9=192 p99.99=10240  MAX=13917 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=803  deferred_reclaims=0  deferred_unmaps=12  pending_bytes=0
    (pressure: 803 mapping(s) expected — the > LRC_HI band cannot be cached)
    == PASSED ==

`--xt 2000000` output:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke reps=4 iters=200 interferers=3
    clock overhead: mean=12 max=83 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=667 ns
        OFF malloc             n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=792 ns
        RT  1st-touch          n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=125 ns
        OFF 1st-touch          n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=84 ns
        RT  free               n=51200      mean=14      p50=0 p99=96 p99.9=320  MAX=1833 ns
        OFF free               n=51200      mean=13      p50=0 p99=42 p99.9=256  MAX=417 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=14      p50=0 p99=42 p99.9=96  MAX=10916 ns
        OFF malloc             n=25600      mean=12      p50=0 p99=42 p99.9=96  MAX=250 ns
        RT  1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=10625 ns
        OFF 1st-touch          n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  free               n=25600      mean=19      p50=0 p99=256 p99.9=384  MAX=9375 ns
        OFF free               n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=10834 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=56      p50=41 p99=384  MAX=542 ns
        OFF malloc             n=6400       mean=19      p50=0 p99=42  MAX=542 ns
        RT  1st-touch          n=6400       mean=12      p50=0 p99=42  MAX=791 ns
        OFF 1st-touch          n=6400       mean=13      p50=0 p99=42  MAX=1000 ns
        RT  free               n=6400       mean=26      p50=0 p99=256  MAX=12125 ns
        OFF free               n=6400       mean=16      p50=0 p99=42  MAX=125 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=122     p50=96 p99=448  MAX=6875 ns
        OFF malloc             n=1600       mean=99      p50=96 p99=192  MAX=5250 ns
        RT  1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=625 ns
        OFF 1st-touch          n=1600       mean=17      p50=0 p99=42  MAX=5792 ns
        RT  free               n=1600       mean=139     p50=96 p99=512  MAX=10791 ns
        OFF free               n=1600       mean=93      p50=96 p99=192  MAX=250 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=2000000    mean=20      p50=0 p99=42 p99.9=192 p99.99=512 p99.999=1792  MAX=14084 ns
        OFF free               n=2000000    mean=20      p50=0 p99=42 p99.9=224 p99.99=7168 p99.999=10240  MAX=12083 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=2  deferred_reclaims=0  deferred_unmaps=0  pending_bytes=0
    FAILED: the realtime arm made 2 new mapping(s) despite prewarming
    == FAILED ==

### 3.1 Re-measured with the harness and `rt_drain` fixed (2026-10-11)

Two changes (`RT_READINESS.md` G7 Result 1b):

- **Harness:** `kame_pool_rt_drain()` in the trough of every iteration — the
  control cycle of `kame_pool.h`'s loop — instead of once per round; FAILED
  on `rt_forced_releases() != 0`, and under `--pressure` on any mapping beyond
  the > LRC_HI band's expected ones (the branch used to accept any count).
- **Library:** `kame_pool_rt_drain()` settles only the parked unmaps.  It also
  emptied this thread's L1 and the global recycle cache — the working set
  `kame_pool_prewarm()` built — so the section after every drain mapped
  again.  That was §3's `rt_violations=2` (`--rt-os-policy 3` aborts in the
  8 MiB band: "large-tier mmap"), and with a per-cycle drain the new harness
  against the old library reports 1,600 violations.  Emptying the caches is
  `kame_pool_trim()` now.

| Path | realtime | default |
|---|---:|---:|
| free, > 256 MiB band — median | **192 ns** | 16,384 ns |
| free, > 256 MiB band — max | **709 ns** | 54,208 ns |
| free, 32 B cross-thread — p99.9 | **128 ns** | 224 ns |
| free, 32 B cross-thread — p99.99 | **192 ns** | 7,168 ns |
| free, 32 B cross-thread — p99.999 | **4,096 ns** | 10,240 ns |

The > 256 MiB rows are the median of three `--pressure` runs (realtime p50
224 / 128 / 192 ns, max 11,375 / 416 / 709 ns; default p50 16,384 ns in all
three, max 73,084 / 49,042 / 54,208 ns).  In each, `deferred_unmaps=800` of
800, `forced_releases=0`, and `rt_violations` = exactly the 800 expected band
mappings.  The default (smoke) run and `--xt 2000000` report
`rt_violations=0`, also under `--rt-os-policy 3`.  All PASSED.  Full-tree
ctest 45/45 (2× Release, 2× -O2 -g with asserts).

First `--pressure` run:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke +pressure reps=4 iters=200 interferers=3
    pressure: added the > LRC_HI band, where the recycle cache is bypassed and frees really munmap
    clock overhead: mean=15 max=1500 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=14      p50=0 p99=42 p99.9=42  MAX=11291 ns
        OFF malloc             n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=333 ns
        RT  1st-touch          n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=17791 ns
        OFF 1st-touch          n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  free               n=51200      mean=15      p50=0 p99=96 p99.9=128  MAX=20875 ns
        OFF free               n=51200      mean=14      p50=0 p99=42 p99.9=128  MAX=7917 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=14      p50=0 p99=42 p99.9=96  MAX=8375 ns
        OFF malloc             n=25600      mean=13      p50=0 p99=42 p99.9=96  MAX=250 ns
        RT  1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=750 ns
        OFF 1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  free               n=25600      mean=16      p50=0 p99=96 p99.9=128  MAX=9416 ns
        OFF free               n=25600      mean=14      p50=0 p99=42 p99.9=96  MAX=250 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=41      p50=41 p99=256  MAX=333 ns
        OFF malloc             n=6400       mean=32      p50=41 p99=224  MAX=416 ns
        RT  1st-touch          n=6400       mean=13      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=6400       mean=12      p50=0 p99=42  MAX=83 ns
        RT  free               n=6400       mean=18      p50=0 p99=128  MAX=292 ns
        OFF free               n=6400       mean=18      p50=0 p99=42  MAX=250 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=133     p50=96 p99=448  MAX=12666 ns
        OFF malloc             n=1600       mean=123     p50=96 p99=512  MAX=625 ns
        RT  1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        RT  free               n=1600       mean=125     p50=96 p99=512  MAX=667 ns
        OFF free               n=1600       mean=121     p50=96 p99=512  MAX=625 ns

      300 MiB (> LRC_HI)
        RT  malloc             n=800        mean=3407    p50=3584  MAX=20792 ns
        OFF malloc             n=800        mean=3244    p50=3072  MAX=43583 ns
        RT  1st-touch          n=800        mean=627     p50=640  MAX=12667 ns
        OFF 1st-touch          n=800        mean=604     p50=640  MAX=8333 ns
        RT  free               n=800        mean=265     p50=224  MAX=11375 ns
        OFF free               n=800        mean=15685   p50=16384  MAX=73084 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=120000     mean=22      p50=41 p99=96 p99.9=192 p99.99=224  MAX=7417 ns
        OFF free               n=120000     mean=21      p50=0 p99=42 p99.9=224 p99.99=10240  MAX=13000 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=800  deferred_reclaims=0  deferred_unmaps=800  forced_releases=0  pending_bytes=0
    (pressure: 800 mapping(s) expected — the > LRC_HI band cannot be cached)
    == PASSED ==

`--xt 2000000`:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke reps=4 iters=200 interferers=3
    clock overhead: mean=11 max=83 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=38333 ns
        OFF malloc             n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=167 ns
        RT  1st-touch          n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=84 ns
        OFF 1st-touch          n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  free               n=51200      mean=13      p50=0 p99=42 p99.9=96  MAX=292 ns
        OFF free               n=51200      mean=13      p50=0 p99=42 p99.9=96  MAX=167 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=208 ns
        OFF malloc             n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=2875 ns
        RT  1st-touch          n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=7208 ns
        OFF 1st-touch          n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=167 ns
        RT  free               n=25600      mean=14      p50=0 p99=42 p99.9=96  MAX=19667 ns
        OFF free               n=25600      mean=13      p50=0 p99=42 p99.9=96  MAX=125 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=18      p50=0 p99=42  MAX=167 ns
        OFF malloc             n=6400       mean=18      p50=0 p99=42  MAX=167 ns
        RT  1st-touch          n=6400       mean=12      p50=0 p99=42  MAX=84 ns
        OFF 1st-touch          n=6400       mean=12      p50=0 p99=42  MAX=83 ns
        RT  free               n=6400       mean=16      p50=0 p99=42  MAX=167 ns
        OFF free               n=6400       mean=16      p50=0 p99=42  MAX=167 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=94      p50=96 p99=384  MAX=500 ns
        OFF malloc             n=1600       mean=80      p50=96 p99=192  MAX=209 ns
        RT  1st-touch          n=1600       mean=12      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=1600       mean=11      p50=0 p99=42  MAX=42 ns
        RT  free               n=1600       mean=98      p50=96 p99=384  MAX=500 ns
        OFF free               n=1600       mean=84      p50=96 p99=224  MAX=333 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=2000000    mean=20      p50=0 p99=42 p99.9=128 p99.99=192 p99.999=4096  MAX=11959 ns
        OFF free               n=2000000    mean=20      p50=0 p99=42 p99.9=224 p99.99=7168 p99.999=10240  MAX=15500 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=0  deferred_reclaims=0  deferred_unmaps=0  forced_releases=0  pending_bytes=0
    == PASSED ==

## 4. Re-measured at `e0974e402` (2026-10-13)

All three sets again, on KAME master `896bc0bec` — kamepoolalloc as of
`e0974e402`: the thread lists (`5fc34e052`), the per-thread cap (`f9e5cb9b3`
… `9f6370a52`) and the owner free's bucket read off `p` (`78e57424c`,
`e0974e402`).  Same machine, OS, toolchain (Apple clang 21.0.0, CMake
3.31.12, Release), comparator builds (sha256 as above) and commands, run
from a worktree of that commit.  Not idle, as before: Chrome Remote
Desktop's host was running and WindowServer used about a third of a core;
load average 2.3–3.5 at the start of each set (the 10-thread benches took it
to 5.7).  2026-10-13 12:13–12:28 UTC.

### 4.1 `bench_compare.sh`

    cmake -S kamepoolalloc/tests -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j10
    bash kamepoolalloc/tests/bench/bench_compare.sh --build-dir build --runs 7 \
        --mimalloc ~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --jemalloc ~/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib

Run twice back to back; the README has the first.

kamepoolalloc bench_compare: HEAD=896bc0bec  host=dhcp-029-018.issp.u-tokyo.ac.jp/arm64  runs=7  os=Darwin

#### 1T (median of 7, M ops/s)

| size      |  system | mimalloc | jemalloc |     kame |
|-----------|---------|----------|----------|----------|
| 64B       |     160 |     1054 |      166 |     1110 |
| 1 KiB     |     133 |      766 |      165 |      765 |
| 16 KiB    |     136 |      353 |      103 |      792 |
| 64 KiB    |      28 |      358 |       29 |      194 |
| 256 KiB   |      28 |      357 |       29 |      177 |
| 1 MiB     |      28 |        8 |       29 |      163 |
| 4 MiB     |      47 |        8 |       29 |      148 |

#### 4 processes (aggregate, M ops/s, median of 7)

| size      |  system | mimalloc | jemalloc |     kame |
|-----------|---------|----------|----------|----------|
| 64B       |     601 |     3989 |      613 |     4256 |
| 16 KiB    |     527 |     1365 |      384 |     3063 |
| 64 KiB    |     106 |     1357 |      109 |      741 |
| 1 MiB     |     106 |       24 |      108 |      627 |

The second run's kame column: single thread 64 B 1134, 1 KiB 749, 16 KiB 829,
64 KiB 193, 256 KiB 175, 1 MiB 160, 4 MiB 149; 4 processes 64 B 4257,
16 KiB 3043, 64 KiB 752, 1 MiB 630.  Its comparators are within 3 % of the
first run's, except mimalloc single-thread 64 KiB (321 against 358) and
system 4-process 64 B (646 against 601).

Against §1 the comparators reproduce to within 2 %, except system
single-thread 16 KiB (136 against 142) and 4-process 64 B (601 against 641).
kame: 64 B 0.98×, 1 KiB 0.95×, 16 KiB 0.89× single thread, 4-process 16 KiB
0.92×; the other rows within 3 %.  §4.2 says where the 1 KiB / 16 KiB step
comes from.

### 4.2 `loop`, interleaved against v1.2.1 and the two commits before `e0974e402`

    kamepoolalloc/tests/bench/bench_ab.sh --tags tree \
        --arm e2c9e6dde=~/bench-ab/arms/e2c9e6dde/kamepoolalloc/out/libkamepoolalloc.dylib \
        --arm 6ae486851=~/bench-ab/arms/6ae486851/kamepoolalloc/out/libkamepoolalloc.dylib \
        --arm v121=~/bench-ab/arms/v121/kamepoolalloc/out/libkamepoolalloc.dylib \
        --arm mi=~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --only loop --reps 9 --work ~/bench-ab-readme --mbench ~/bench-ab/mimalloc-bench

Arms: tree = `896bc0bec` (sha256 `398c663ba240eb36`); `e2c9e6dde`, just
before `78e57424c` (sha256 `ead680cd7452ea6a`); `6ae486851`, after
`78e57424c` and before `e0974e402` (sha256 `d9de8bedb858cfe4`); v121 =
`45f1d07f9`, the v1.2.1 tag (sha256 `15903cd5c08a8c1f`).  The two new arms
are `git archive <commit> kamepoolalloc` built as `bench_ab.sh` builds a tag
(`cmake -DCMAKE_BUILD_TYPE=Release -DKAMEPOOLALLOC_BUILD_TESTS=OFF`).  v121
stands in for §1's `2e8f940fd`: what lies between them is the calloc /
`posix_memalign` interposition, the `rt_drain` change and a comment, none of
them on `bench_loop`'s malloc / free path.  Results
`~/bench-ab-readme/results-dhcp-029-018-20261013-211714`:

| bench | config | metric | kp-tree | e2c9e6dde | 6ae486851 | v121 | mi | e2c9e6dde / kp-tree | 6ae486851 / kp-tree | v121 / kp-tree | mi / kp-tree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loop | 64B | Mops ↑ | 1130 (1119–1137) | 1132 (1097–1137) | 1130 (1111–1137) | 1130 (1097–1134) | 1058 (979–1081) | 1.00× | 1.00× | 1.00× | 0.94× |
| loop | 1024B | Mops ↑ | 772 (746–787) | 754 (741–810) | 763 (740–812) | 798 (780–840) | 777 (756–794) | 0.98× | 0.99× | 1.03× | 1.01× |
| loop | 16384B | Mops ↑ | 775 (766–800) | 771 (728–814) | 796 (753–841) | 840 (772–908) | 363 (338–374) | 0.99× | 1.03× | 1.08× | 0.47× |

- `78e57424c` and `e0974e402` are level on the M5 — tree against
  `e2c9e6dde` 1.00× / 1.02× / 1.01× — the `loop` re-run
  `THREAD_LIST_CHECK.md` asked for after the Windows run.
- Against v1.2.1 the tree is 1.00× at 64 B, 0.97× at 1 KiB and 0.92× at
  16 KiB: the cost of the per-thread cap that `THREAD_LIST_CHECK.md`
  measured on this machine (pre / tree 1.03–1.12× at 1 KiB and 1.08–1.12×
  at 16 KiB in its M5 runs).  §4.1's step against §1 is that, not drift: the comparators
  did not move.
- 1 KiB is level with mimalloc here (1.01×), where §4.1's single run has
  kame 765 against 766.

### 4.3 mimalloc-bench (`bench_ab.sh`)

    kamepoolalloc/tests/bench/bench_ab.sh --tags tree --arm sys= \
        --arm mi=~/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib \
        --arm je=~/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib \
        --only "xmalloc larson rptest" --threads 10 --reps 5 \
        --work ~/bench-ab-readme --mbench ~/bench-ab/mimalloc-bench

The §2 command with a fresh work directory (`~/bench-ab/kp-tree` was
configured from the main checkout).  Every run produced a result; rptest now
measures kame (§2.1).

| bench | config | metric | kp-tree | sys | mi | je | sys / kp-tree | mi / kp-tree | je / kp-tree |
|---|---|---|---|---|---|---|---|---|---|
| xmalloc | w10,64B | Mfree ↑ | 498 (490–498) | 257 (236–264) | 516 (508–518) | 408 (407–409) | 0.52× | 1.04× | 0.82× |
| larson | t10 | Mops ↑ | 294 (289–308) | 26.2 (25.7–26.9) | 217 (204–219) | 421 (412–425) | 0.09× | 0.74× | 1.43× |
| rptest | t10 | Mops_cpu ↑ | 6.46 (5.86–6.58) | 2.88 (2.72–3) | 5.66 (5.47–5.89) | 5.7 (5.55–5.82) | 0.45× | 0.88× | 0.88× |

Against §2 / §2.1 the comparators are within 3 % on xmalloc and larson; on
rptest system is +9 % (2.88 against 2.64) and jemalloc −8 % (5.70 against
6.20).  kame: xmalloc 489 → 498, larson 267 → 294, rptest 6.29 → 6.46.

`provenance.txt`:

    date      2026-10-13T12:17:37Z
    host      dhcp-029-018.issp.u-tokyo.ac.jp
    kernel    Darwin 27.0.0 arm64
    cpu       Apple M5 Ultra, 10P + 20E, 96 GiB, Mac17,15
    os        macOS 27.0.1 (26A434)
    power     Now drawing from 'AC Power'
    benches   /Users/ssp/KAME/.claude/worktrees/recursing-einstein-777aa1/kamepoolalloc @ 896bc0bec, /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ (Apple clang version 21.0.0 (clang-2100.3.34.2))
    arm       kp-tree: /Users/ssp/bench-ab-readme/kp-tree/out/libkamepoolalloc.dylib (684232 B, sha256 398c663ba240eb36, tree /Users/ssp/KAME/.claude/worktrees/recursing-einstein-777aa1/kamepoolalloc at 896bc0bec, built by /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++)
    arm       sys: no preload (system malloc)
    arm       mi: /Users/ssp/bench-ab/alloc/mimalloc-3.3.2/build/libmimalloc.dylib (233072 B, sha256 5153705f94d7d8b4)
    arm       je: /Users/ssp/bench-ab/alloc/jemalloc-5.3.0/lib/libjemalloc.2.dylib (649248 B, sha256 9290fc4e3540a8f2)
    mbench    /Users/ssp/bench-ab/mimalloc-bench @ 69c41ed, binaries in /Users/ssp/bench-ab-readme/mbench-build
    params    reps=5 threads="10" only="xmalloc larson rptest" quick=0 xlat="1024x200000 1024x1000000 3000x200000" xlat_cpu=none loop_iters=20000000 frame_frames=5000

### 4.4 RT (`bench_rt_wcet`)

    build/bench_rt_wcet --pressure        # three times
    build/bench_rt_wcet --xt 2000000      # three times

The §3.1 harness and library behaviour, from the §4.1 build.  Every run
PASSED.  Each `--pressure` run: `deferred_unmaps=800` of 800,
`forced_releases=0`, `rt_violations=800` (exactly the band's expected
mappings).  Each `--xt` run: `rt_violations=0`.  The README's rows, median of
the three runs:

| Path | realtime | default |
|---|---:|---:|
| free, > 256 MiB band — median | **192 ns** | 14,336 ns |
| free, > 256 MiB band — max | **417 ns** | 49,208 ns |
| free, 32 B cross-thread — p99.9 | **96 ns** | 224 ns |
| free, 32 B cross-thread — p99.99 | **192 ns** | 7,168 ns |
| free, 32 B cross-thread — p99.999 | **7,168 ns** | 10,240 ns |

> 256 MiB band, per run: realtime p50 128 / 192 / 192 ns, max 333 / 459 /
417 ns; default p50 16,384 / 14,336 / 14,336 ns, max 49,208 / 44,916 /
131,458 ns.  §3.1's realtime max of 11.4 µs in one run did not recur.
Cross-thread: realtime p99.9 96 and p99.99 192 ns in all three, p99.999
7,168 / 3,072 / 7,168 ns (§3.1: 4,096); default 224 / 7,168 / 10,240 ns in all
three.

First `--pressure` run:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke +pressure reps=4 iters=200 interferers=3
    pressure: added the > LRC_HI band, where the recycle cache is bypassed and frees really munmap
    clock overhead: mean=14 max=84 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=8791 ns
        OFF malloc             n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  1st-touch          n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=84 ns
        OFF 1st-touch          n=51200      mean=12      p50=0 p99=42 p99.9=42  MAX=5375 ns
        RT  free               n=51200      mean=14      p50=0 p99=42 p99.9=96  MAX=7333 ns
        OFF free               n=51200      mean=14      p50=0 p99=42 p99.9=96  MAX=6541 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=14      p50=0 p99=42 p99.9=96  MAX=7458 ns
        OFF malloc             n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=125 ns
        RT  1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=1125 ns
        OFF 1st-touch          n=25600      mean=12      p50=0 p99=42 p99.9=42  MAX=84 ns
        RT  free               n=25600      mean=15      p50=0 p99=42 p99.9=128  MAX=10416 ns
        OFF free               n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=125 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=19      p50=0 p99=42  MAX=208 ns
        OFF malloc             n=6400       mean=19      p50=0 p99=42  MAX=166 ns
        RT  1st-touch          n=6400       mean=12      p50=0 p99=42  MAX=84 ns
        OFF 1st-touch          n=6400       mean=13      p50=0 p99=42  MAX=83 ns
        RT  free               n=6400       mean=17      p50=0 p99=42  MAX=125 ns
        OFF free               n=6400       mean=17      p50=0 p99=42  MAX=167 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=88      p50=96 p99=224  MAX=292 ns
        OFF malloc             n=1600       mean=87      p50=96 p99=192  MAX=250 ns
        RT  1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        RT  free               n=1600       mean=96      p50=96 p99=224  MAX=7167 ns
        OFF free               n=1600       mean=93      p50=96 p99=224  MAX=292 ns

      300 MiB (> LRC_HI)
        RT  malloc             n=800        mean=2948    p50=3072  MAX=12917 ns
        OFF malloc             n=800        mean=2789    p50=3072  MAX=9375 ns
        RT  1st-touch          n=800        mean=575     p50=512  MAX=4000 ns
        OFF 1st-touch          n=800        mean=571     p50=512  MAX=4042 ns
        RT  free               n=800        mean=133     p50=128  MAX=333 ns
        OFF free               n=800        mean=14825   p50=16384  MAX=49208 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=120000     mean=19      p50=0 p99=42 p99.9=96 p99.99=192  MAX=7667 ns
        OFF free               n=120000     mean=20      p50=0 p99=42 p99.9=192 p99.99=8192  MAX=10792 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=800  deferred_reclaims=0  deferred_unmaps=800  forced_releases=0  pending_bytes=0
    (pressure: 800 mapping(s) expected — the > LRC_HI band cannot be cached)
    == PASSED ==

First `--xt 2000000` run:

    == bench_rt_wcet (§75 / G7) ==
    backend: kame_pool_malloc/free (direct)
    mode=smoke reps=4 iters=200 interferers=3
    clock overhead: mean=17 max=9667 ns
    measured-thread priority: QOS_CLASS_USER_INTERACTIVE

      64 B    (bucket)
        RT  malloc             n=51200      mean=14      p50=0 p99=42 p99.9=42  MAX=125 ns
        OFF malloc             n=51200      mean=14      p50=0 p99=42 p99.9=42  MAX=6583 ns
        RT  1st-touch          n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=84 ns
        OFF 1st-touch          n=51200      mean=13      p50=0 p99=42 p99.9=42  MAX=8000 ns
        RT  free               n=51200      mean=14      p50=0 p99=42 p99.9=96  MAX=167 ns
        OFF free               n=51200      mean=15      p50=0 p99=42 p99.9=128  MAX=10500 ns

      4 KiB   (bucket)
        RT  malloc             n=25600      mean=14      p50=0 p99=42 p99.9=42  MAX=208 ns
        OFF malloc             n=25600      mean=14      p50=0 p99=42 p99.9=42  MAX=625 ns
        RT  1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=84 ns
        OFF 1st-touch          n=25600      mean=13      p50=0 p99=42 p99.9=42  MAX=3792 ns
        RT  free               n=25600      mean=14      p50=0 p99=42 p99.9=42  MAX=167 ns
        OFF free               n=25600      mean=16      p50=0 p99=42 p99.9=128  MAX=34792 ns

      256 KiB (dedicated)
        RT  malloc             n=6400       mean=20      p50=0 p99=42  MAX=167 ns
        OFF malloc             n=6400       mean=30      p50=41 p99=192  MAX=416 ns
        RT  1st-touch          n=6400       mean=13      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=6400       mean=13      p50=0 p99=42  MAX=42 ns
        RT  free               n=6400       mean=18      p50=0 p99=42  MAX=208 ns
        OFF free               n=6400       mean=18      p50=0 p99=96  MAX=250 ns

      8 MiB   (large)
        RT  malloc             n=1600       mean=87      p50=96 p99=224  MAX=334 ns
        OFF malloc             n=1600       mean=106     p50=96 p99=448  MAX=583 ns
        RT  1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        OFF 1st-touch          n=1600       mean=13      p50=0 p99=42  MAX=42 ns
        RT  free               n=1600       mean=90      p50=96 p99=224  MAX=333 ns
        OFF free               n=1600       mean=108     p50=96 p99=448  MAX=625 ns

      32 B cross-thread free (producer allocs, we free)
        RT  free               n=2000000    mean=19      p50=0 p99=42 p99.9=96 p99.99=192 p99.999=7168  MAX=57958 ns
        OFF free               n=2000000    mean=20      p50=0 p99=42 p99.9=224 p99.99=7168 p99.999=10240  MAX=13542 ns
        (OFF batches CAP=1024 then one free pays for the whole buffer; RT takes push_direct)

    rt_violations=0  deferred_reclaims=0  deferred_unmaps=0  forced_releases=0  pending_bytes=0
    == PASSED ==
