# mimalloc-bench PR draft — adding `kp` (kamepoolalloc)

Status: prepared 2026-06-11.  Prerequisites all met:
* standalone repo: https://github.com/northriv/kamepoolalloc (subtree mirror
  of `KAME/kamepoolalloc/`, synced via `git subtree split`)
* pinned tag: `v1.0.0`
* top-level CMake builds `out/libkamepoolalloc.so` with the full malloc
  interpose default-on for `LD_PRELOAD` use
* local-suite soak (this container, glibc/x86-64): cfrac, espresso, barnes,
  alloc-test 1/N, larson, larson-sized, xmalloc-test, cache-thrash,
  cache-scratch, malloc-large, mstress, mleak 10/100, rptest, glibc-simple,
  glibc-thread — all complete, no crash / hang / RSS blow-up.
  (sh6/sh8bench skipped here: microquill.com download blocked; redis / lean /
  gs / lua / rocksdb need their extra setup — re-soak with the full
  `build-bench-env.sh` flow on an open-network Linux box before submitting.)

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

* Re-soak via the unmodified `./build-bench-env.sh kp bench` +
  `./bench.sh kp allt` on a normal-network Linux host (covers redis, lean,
  sh6/sh8bench and the suite's own result parsing).
* The activation banner ("Reserve swap space ... ") goes to **stderr** only
  (documented sanity check in `allocator.h`); it does not disturb the
  suite's stdout parsing.  If upstream prefers silence, gate it behind an
  env var in a follow-up tag.
* `out/libkamepoolalloc.so` is a symlink to `libkamepoolalloc.so.8`
  (SOVERSION); `LD_PRELOAD` through the symlink is fine.
