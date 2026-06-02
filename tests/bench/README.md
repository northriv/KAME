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
      kame       2.7  (override)
      kame       3.0  (direct, bench_xthread_pool)
      mi3       82
      je        11.1

The override path adds ~10% overhead over direct (240 vs 268 in the loop
case).  Cross-thread free is the actual bottleneck: kame is currently 5-10x
slower than libc on small same-class cross-thread frees and ~25x slower
than mimalloc on the same.  This is a known design trade-off — cross-thread
free workloads are uncommon in kame's primary use case (STM Payload
copy-on-write, instrument drivers, scientific compute), where allocation
and release are dominated by same-thread patterns.

### Size-class cliff observed at workers=2 (kame_pool_malloc direct)

    size=    8B  free/sec: 28.42 M     ← beats libc + jemalloc
    size=   32B  free/sec: 13.91 M     ← parity with libc
    size=   64B  free/sec:  3.15 M     ← cliff (5x slower than libc)
    size=  256B  free/sec: 12.06 M     ← parity with libc
    size= 1024B  free/sec:  3.49 M     ← cliff
    size= 4096B  free/sec:  1.79 M

The dip is workload-specific (cross-thread free × certain size classes),
not a general weakness.  Sizes 8 / 32 / 256 are competitive even on this
worst-case pattern; 64 / 1024 hit producer-consumer cacheline contention
on the chunk's `m_flags` bitmap word.

## Comparing against mimalloc-bench's binary

To cross-check against the published `xmalloc-test`:

    git clone --depth 1 https://github.com/daanx/mimalloc-bench.git
    cc -O2 -pthread -o xmalloc-test \
       mimalloc-bench/bench/xmalloc-test/xmalloc-test.c

It implements the identical workload to `bench_xthread`.  Numbers should
match within run-to-run noise.
