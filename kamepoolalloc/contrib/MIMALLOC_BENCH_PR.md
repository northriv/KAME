# mimalloc-bench PR draft — adding `kp` (kamepoolalloc)

Status: prepared 2026-06-11.  Upstream since: added in daanx/mimalloc-bench
#265 (merged 2026-09-02, pinned v1.0.2), pin moved to v1.1.0 in #270 (merged
2026-09-02); v1.3.0 is the next pin (see Notes).  Prerequisites all met:
* standalone repo: https://github.com/northriv/kamepoolalloc (subtree mirror
  of `KAME/kamepoolalloc/`, synced via `git subtree split`)
* pinned tag: v1.3.0+ (see Notes — v1.2.0 runs larson at 0.4–0.7x of
  v1.1.0, tags up to v1.1.0 carry a thread-exit use-after-free, tags before
  v1.1.0 also a double-allocation defect, and v1.0.0 predates the banner
  gating and the Linux `malloc_usable_size` co-interpose)
* top-level CMake builds `out/libkamepoolalloc.so` with the full malloc
  interpose default-on for `LD_PRELOAD` use
* **Full-suite soak complete** (glibc/x86-64, 4-core container,
  2026-06-11): the 17 local benches (cfrac, espresso, barnes,
  alloc-test 1/N, larson, larson-sized, xmalloc-test, cache-thrash,
  cache-scratch, malloc-large, mstress, mleak 10/100, rptest,
  glibc-simple, glibc-thread) plus gs, lua, lean (stdlib compile),
  redis 6.2.7 (387.6 k rps vs glibc 378.8 k), and sh6bench / sh8bench
  (genuine microquill sources, SHA256-verified against upstream's pins;
  kame 4.4× / 5.4× vs glibc at 8 threads) — all complete, no crash /
  hang / RSS blow-up.  Only rocksdb (optional, extra setup) was skipped.
* The soak CAUGHT AND FIXED one release blocker: redis-server SEGV'd
  because the Linux strong-symbol family lacked a `malloc_usable_size`
  co-interpose (glibc walked its own heap metadata on a pool pointer).
  Fixed in `allocator.cpp` (dlsym RTLD_NEXT forward for foreign
  pointers); redis now passes and beats glibc.

The upstream README invites this: "It is quite easy to add new benchmarks
and allocator implementations -- please do so!"

## Patch (3 hunks)

### 1. `build-bench-env.sh` — version pin (in the version block)

```sh
readonly version_kp=v1.3.0
```

### 2. `build-bench-env.sh` — flag plumbing + help + build section

Flag default / `all` expansion / `case` arm follow the existing pattern
(`setup_kp=0`, etc.).  Help line:

```sh
        echo "  kp                           setup kamepoolalloc ($version_kp)"
```

Build section (after e.g. the `setup_rp` block; `checkout` is the
upstream helper — clones into `$devdir/kp` and leaves us in it):

```sh
if test "$setup_kp" = "1"; then
  checkout kp $version_kp https://github.com/northriv/kamepoolalloc
  cmake -B out -DCMAKE_BUILD_TYPE=Release
  cmake --build out --parallel $procs
  popd
fi
```

### 3. `bench.sh` — allocator registry

```sh
alloc_lib_add "kp"     "$localdevdir/kp/out/libkamepoolalloc$extso"
```

and add `kp` to the `alloc_all` list.

## PR description sketch

> Add kamepoolalloc (`kp`) — a lock-free four-tier pool allocator
> (1 B .. multi-GiB: buckets / dedicated chunks / large mmap / huge) with a
> per-thread two-level recycle cache for the 32 KiB .. 32 MiB range.
> Dual-licensed Apache-2.0 OR GPL-2.0+.  Production allocator of the KAME
> instrument-control framework since 2008; chunk-claim / recycle / orphan-chain
> protocols are TLA+ and GenMC (RC11) model-checked.
> https://github.com/northriv/kamepoolalloc

Keep the PR to the three hunks above — no README table edit (the maintainer
regenerates results himself), no benchmark changes.

## Notes / open items before submitting

* The pin is `v1.3.0`.  v1.2.0 fixed the use-after-free below, but its
  bookkeeping cost speed the suite measures: larson 0.37–0.42x of v1.1.0 on
  Zen 2 (Ohtaka, 4–64 threads) and 0.67–0.74x on Apple M3 / M5, rptest
  0.59–0.84x on Zen 2 at 4–64 threads, and up to nine times the chunks under
  mixed-size churn (`design/V1_2_PERF_HANDOFF.md` §3–§7).  The fix that
  restored v1.1.0's sequence with every walk bounded — larson 1.02–1.14x,
  rptest 0.90–1.00x — was to be v1.2.1 and was never tagged; v1.3.0 carries
  it, and the use-after-free fix.  v1.3.0 then replaces the bounded walks
  with per-thread free lists (README §thread-list).  Against that fix:
  larson 1.08–1.15x, rptest 1.00–1.06x and xmalloc-test 1.08–1.14x on an
  Intel Xeon W under macOS, at v1.3.0's code; larson 1.10–1.11x and rptest
  1.10–1.12x on the M5 Ultra at the first thread-list commit, before the
  per-bucket cap, which costs larson up to 4 % there
  (`design/THREAD_LIST_CHECK.md`).  On Zen 2 (Ohtaka, v1.3.0's code) larson
  is 1.16–1.26x of that fix on the same node, ahead of mimalloc at 4–16
  threads, and bench_loop 1.16x / 1.26x / 1.67x at 64 B / 1 KiB / 16 KiB;
  alloc_tune_report lost two cells there, 1 MiB at 4 threads (0.82x) and
  8 MiB single-thread (0.87x), neither profiled yet
  (`design/README_BENCH_OHTAKA.md` §5).  It also makes
  `kame_calloc` skip the memset of a fresh large-tier mapping (Linux and
  macOS), and pools `calloc` / `posix_memalign` / `aligned_alloc` in the
  macOS dylib, which matters to nobody running the suite on Linux.  Do not
  pin v1.2.0.
* Every tag before v1.2.0 carries a thread-exit use-after-free: a free
  that returned a slot to another thread's chunk read a pointer into the
  owner's TLS, returned the slot, and only then stored through the pointer,
  and an owner that exited in between had that TLS freed under the store.
  On glibc the store faults when the dead thread's stack has already been
  unmapped — `alloc_tsd_exclusivity_test`: 22–34 crashes in 2000 runs on the
  code before the fix, 0 in 2000 after (`design/POOL_FL_AVAIL_LINUX_CHECK.md` §6–§7) —
  and about a hundred times as often it lands silently, in a stack glibc
  still caches (`design/FORCE_WALK_TOCTOU_HANDOFF.md` §3a).  macOS has never
  crashed on it, which clears nothing: the code is the same, and in the
  dylib a dead thread's TLS block is freed heap, where a write does not
  fault.
  v1.2.0's §revive removed the store altogether: no free touches another
  thread's TLS.  CI is green on either pin, and a silent hit would not turn
  it red — which is why the pin has to move rather than wait.
* `v1.1.0`, the previous floor, closed an older defect that later tags keep
  closed.  Every tag before it can
  hand the SAME BLOCK to two live users on Linux: a free arriving from a
  thread that had finished its own allocator teardown went into a destroyed
  cross-dealloc batch, and a slot returned to the bitmap after its owner had
  reissued it.  pthread_key destructors that free run after the C++
  thread_local ones, so any preloaded program with such a key is exposed —
  which is most of them.  Measured 17 failures in 44 runs of the reproducer
  it was found with, 0 in 44 after; a benchmark suite would see it as
  unexplained corruption, not as a slow allocator.
  `v1.1.0` also keeps what the older pin was chosen for: the dylib banner
  gating and the Linux `malloc_usable_size` co-interpose (v1.0.0 predates
  both — its redis run would crash), plus the word-cache FS=true cold path
  (default ON).
* Optional: one `./bench.sh kp allt` pass on a normal-network Linux host
  to exercise the suite's own result-parsing wrappers (the benches
  themselves are all soaked); rocksdb if desired.
* sh6bench/sh8bench sources are proprietary (microquill) — never commit
  them to this repo; the suite downloads them itself.
* The activation banner ("Reserve swap space ... ") is **silent in the
  dylib build** (gated on `KAMEPOOLALLOC_DYLIB`; re-enable with
  `KAME_POOL_VERBOSE=1`), so the suite sees a quiet allocator.  Present from
  v1.0.1; the pin above is later than that for a different reason.
* `out/libkamepoolalloc.so` is a symlink to `libkamepoolalloc.so.8`
  (SOVERSION); `LD_PRELOAD` through the symlink is fine.
