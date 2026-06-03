# kamepoolalloc/tests/bench — third-party-style allocator microbenches

Reproducible local copies of the workloads that mimalloc-bench / glibc-bench
exercise, written so they can be linked against `libkamepoolalloc.so`
(`LD_PRELOAD` strong-symbol override path) **or** call `kame_pool_malloc` /
`kame_pool_free` directly (bypass-override path).  This split lets you
attribute any throughput delta to either:

  * the **malloc/free strong-symbol override layer** (Linux always-on, macOS
    opt-in via `KAMEPOOLALLOC_FULL_INTERCEPT`, Windows via §31 IAT redirect), or
  * the **pool allocator core** (chunked bitmap freelist, cross-thread batch).

Build (with the parent `kamepoolalloc/tests/` CMakeLists):

    cd kamepoolalloc/tests/build-native
    cmake .. && make -j

Targets produced (all under `bench/`):

| Binary               | Workload              | Allocator route         |
|----------------------|-----------------------|-------------------------|
| `bench_loop`         | single-thread hot     | `malloc`/`free`         |
| `bench_xthread`      | producer/consumer x-T | `malloc`/`free`         |
| `bench_xthread_pool` | producer/consumer x-T | `kame_pool_malloc`/free |

`bench_loop` and `bench_xthread` route through `malloc/free`, so they pick
up whatever allocator the dynamic linker resolves first.  Run with

  * `./bench_loop 64 30000000`                      — libc baseline
  * `LD_PRELOAD=…/libkamepoolalloc.so ./bench_loop` — kame (override path)
  * `LD_PRELOAD=…/libmimalloc.so      ./bench_loop` — mimalloc
  * `LD_PRELOAD=…/libjemalloc.so.2    ./bench_loop` — jemalloc

`bench_xthread_pool` always uses the pool directly (no override route).

### Reproducing the README's head-to-head tables

`bench_compare.sh` runs the exact 1T + 4-process matrix that the
`kamepoolalloc/README.md` "Benchmarks" section publishes — kame vs
system / mimalloc / jemalloc across 64 B … 4 MiB, median of N runs,
markdown-formatted so the output pastes straight in:

    cd kamepoolalloc/tests/build && cmake .. && make -j
    ../bench/bench_compare.sh                       # full table, 5 runs each
    ../bench/bench_compare.sh --runs 1 --no-mt      # quick sanity
    ../bench/bench_compare.sh --mimalloc /path --jemalloc /path   # explicit

It auto-detects mimalloc/jemalloc in the usual MacPorts / Homebrew /
`/usr/lib/x86_64-linux-gnu/` paths and falls back to "-" in the column
when one is missing.  Build dir defaults to `tests/build`; override
with `--build-dir`.

## Why these specific patterns

`bench_loop` is the **best case** for any pool allocator: same-thread
alloc-then-free in a tight loop, refilling a 1-slot freelist that lives in
L1d.  Numbers here measure raw per-call cost; this is also what
`alloc_minimal_bench` measures, but through `malloc` instead of `new`.

`bench_xthread` is the **worst case** for per-thread freelist designs: each
object is allocated in thread A, then released in thread B.  The releasing
thread cannot push the slot back to A's TLS freelist (the slot belongs to
A's chunk), so it accumulates the return on a cross-thread batch (kame's
`CrossDeallocBatch`, mimalloc's `_mi_free_block_mt`) and eventually CASes
the bitmap word back.  The producer thread sees the return only after its
own chunk-walk path notices the bitmap change — meanwhile it claims fresh
chunks, inflating the resident set.  This is the workload that
mimalloc-bench's `xmalloc-test` (Lever & Boreham, USENIX 2000) exercises.

## Reference numbers (Linux dev env, x86-64, 4 cores)

    bench_loop 64 30M (M ops/s, higher=better)
      libc     200
      mi3      218
      kame     241  (override)
      kame     268  (direct, via alloc_minimal_bench)
      je       309

    bench_xthread -w2 -s64 -t5 (M free/sec, higher=better)
      libc      15.3
      kame      45    (override + direct, after force_walk fix)
      mi3       82
      je        11.1

The override path adds ~10 % overhead over direct (240 vs 268 in the loop
case).  Cross-thread free is the worst case for any per-thread freelist
design; the production-side fix introduced in the same series as this
README closes the FS=true gap (sizes 16..368) to the point where kame
beats libc by ~3 ×, with mimalloc still ahead on this pattern.  The
remaining FS=false gap (sizes ≥ 1024) is a chunk-recycle issue documented
below.

### Size-class profile at workers=2 (kame_pool_malloc direct)

After the `CrossDeallocBatch::flush sets owner force_walk hint` commit
in this series:

    size=    8B  free/sec: 28.4 M     ← beats libc + jemalloc
    size=   32B  free/sec: 34.1 M     ← beats libc (2.3 ×)
    size=   64B  free/sec: 47.2 M     ← beats libc (3.1 ×); was 3.2 M
                                        before the force_walk fix
    size=  256B  free/sec: 12.2 M     ← parity with libc
    size= 1024B  free/sec:  3.5 M     ← still slow (FS=false large slot)
    size= 4096B  free/sec:  1.8 M     ← still slow (FS=false large slot)

The remaining ≥ 1024 B dip is `madvise(DONTNEED)` + page-fault thrash on
the chunk-recycle path (per `perf record`: 27 % of samples land in kernel
TLB flush / page-zeroing routines when consumer-drained chunks get
released eagerly and the producer immediately needs a fresh chunk).  A
walk-then-release reorder of `allocate_chunk_path` recovers it to ~7 M
free/sec but regresses FS=true sizes 32/64 by 30 %, so the trade-off was
left as documented future work.  Cross-thread free is rare in kame's
primary use case (STM Payload copy-on-write, instrument drivers,
scientific compute) — see CLAUDE.md.

## Comparing against mimalloc-bench's binary

To cross-check against the published `xmalloc-test`:

    git clone --depth 1 https://github.com/daanx/mimalloc-bench.git
    cc -O2 -pthread -o xmalloc-test \
       mimalloc-bench/bench/xmalloc-test/xmalloc-test.c

It implements the identical workload to `bench_xthread`.  Numbers should
match within run-to-run noise.
