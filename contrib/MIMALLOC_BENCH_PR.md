# mimalloc-bench PR draft — adding `kp` (kamepoolalloc)

Status: prepared 2026-06-11.  Prerequisites all met:
* standalone repo: https://github.com/northriv/kamepoolalloc (subtree mirror
  of `KAME/kamepoolalloc/`, synced via `git subtree split`)
* pinned tag: v1.0.1+ (must include the banner gating AND the Linux
  `malloc_usable_size` co-interpose — see Notes)
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
readonly version_kp=v1.0.0
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

* Tag a release containing BOTH the dylib banner gating and the Linux
  `malloc_usable_size` co-interpose, and pin `version_kp` to it — v1.0.0
  predates both (its redis run would crash).
* Optional: one `./bench.sh kp allt` pass on a normal-network Linux host
  to exercise the suite's own result-parsing wrappers (the benches
  themselves are all soaked); rocksdb if desired.
* sh6bench/sh8bench sources are proprietary (microquill) — never commit
  them to this repo; the suite downloads them itself.
* The activation banner ("Reserve swap space ... ") is **silent in the
  dylib build** (gated on `KAMEPOOLALLOC_DYLIB`; re-enable with
  `KAME_POOL_VERBOSE=1`), so the suite sees a quiet allocator.  Pin the
  tag that contains this gating (v1.0.1 or later), not v1.0.0.
* `out/libkamepoolalloc.so` is a symlink to `libkamepoolalloc.so.8`
  (SOVERSION); `LD_PRELOAD` through the symlink is fine.
