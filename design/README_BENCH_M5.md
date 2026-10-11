# The README's benchmarks on the Mac Studio M5 Ultra (2026-10-11)

Re-measurement of the tables in `kamepoolalloc/README.md` "Benchmarks"
(Apple MacBook Air M3) on the Mac Studio M5 Ultra, to replace them.  KAME
master `c9fa20d33` (kamepoolalloc as of `2e8f940fd`, the v1.2.0 regression
fix; the commits since change only tests and design notes).

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
(`kame_pool_set_rt_pending_cap`) for this band.

**`--xt 2000000` ended `FAILED`: `rt_violations=2`** — the realtime arm made
two new mappings despite prewarming.  The measurement itself completed; the
same 2 violations appear on every build with the default / `--full`
arguments, master included, and were noted before this run (ctest's smaller
arguments pass).

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
