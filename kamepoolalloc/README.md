# kamepoolalloc

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20(MinGW)-lightgrey)]()

A lock-free, per-thread, four-tier pool allocator spanning **1 B to 32 MiB** —
bucketed small objects, dedicated mid chunks, and `munmap`-backed large blocks,
all freed through one uniform path.  Born for **multi-threaded STM-style
transactional workloads** but usable as a general-purpose drop-in `new` /
`delete` (or C `malloc`) replacement on macOS, Linux, and Windows (MinGW).

Carved out of the [KAME](https://github.com/northriv/KAME) measurement framework
and **dual-licensed under Apache 2.0 OR GPL-2.0-or-later** at your choice — so
it can be embedded into permissive / proprietary projects (Apache 2.0 path) or
linked into GPLv2-only projects such as KAME itself (GPL path).

## Highlights

- **Lock-free fast path** — TLS freelist pop/push, no atomics on the hot path.
  Hot-path TLS is `initial-exec` (no `__tls_get_addr` thunk).
- **Four allocation tiers, one `free`** — every pointer is resolved in O(1) by
  a 2-level radix tree, so `free`/`realloc`/`malloc_usable_size` work uniformly
  across all tiers:
  - **Buckets** (1 B .. 32 KiB): 51 size classes.  The ALIGN ≥ 1024 tiers are
    *full-usable* (per-slot metadata in a chunk-header side array, not a borrow
    header), so power-of-2 page requests (4/8/16/32 KiB) get an exact,
    page-aligned slot with **0 % round-up**.
  - **Dedicated chunks** (32 KiB .. 4 MiB): one N-unit chunk from a 32 MiB pool
    region.
  - **Large mmap** (4 MiB .. 32 MiB): one 32-MiB-aligned `mmap` per allocation,
    served warm from the recycle cache, **`munmap`'d on free** when not recycled.
  - **Huge mmap** (> 32 MiB): a multi-region `mmap` per allocation, of which only
    the head 32-MiB radix slot is registered — safe because the allocation's sole
    valid pointer resolves to that slot and the tail slots are never standalone
    lookup targets (and the OS keeps the whole span mapped, so no other allocation
    can claim a tail slot's VA).  **`munmap`'d on free**; bypasses the recycle
    cache (its log index tops out at 32 MiB, so a cached huge block could
    over-satisfy a smaller huge request and pin its RSS — the reason
    libc / jemalloc / mimalloc don't pool their huge class).  Plain libc `malloc`
    is reached only if the `mmap` itself fails.
- **Two-level recycle cache** for the large tiers (32 KiB .. 32 MiB): a
  per-thread **L1** (no atomics, ping-pong absorbed) in front of a global
  lock-free **L2** log-slot cache (no working-set cliff, byte-capped).  Reuses
  warm, resident blocks — skips the `mmap`/`madvise`/re-fault cost that makes
  every other allocator fall off a cliff above ~64 KiB.
- **Aligned allocation served from the pool** — `posix_memalign` /
  `aligned_alloc` / `operator new(align_val_t)` up to 4 KiB alignment route to
  the matching bucket (slots are inherently ALIGN-aligned); larger alignments
  use the 256 KiB-aligned dedicated/mmap tiers.  No `_aligned_free` pairing —
  freed by the ordinary `free`.
- **Per-thread DLL chunks** — no global allocator lock, no contention until
  the chunk-claim slow path; cross-thread frees via a holding batch +
  bit-clear coalescing.
- **Standards-conformant OOM** — throwing `operator new` runs the installed
  `std::new_handler` loop then throws `std::bad_alloc`; nothrow / C-API paths
  return `nullptr` + `errno = ENOMEM`.  No `std::terminate` across the noexcept
  C boundary.
- **Bounded VA + prompt RSS** — two independent runtime caps:
  `kame_pool_set_max_bytes()` (fresh-region mmap ceiling) and
  `kame_pool_set_large_cache_cap()` (the large-recycle cache's resident
  footprint, split ~half global L2 / ~half aggregate per-thread L1).
  `madvise(MADV_FREE/DONTNEED)` on chunk release — default also at thread exit,
  toggle via `kame_pool_set_thread_exit_reclaim()`.
- **Verified** — TSAN race-free, UBSAN clean (incl. `vptr`), ASan clean; the
  chunk-claim / chunk-recycle protocol is TLA+ model-checked and the
  large-recycle cache's exclusive-ownership / no-premature-release (UAF /
  double-free) safety is GenMC (RC11) model-checked.  Builds 64-bit and 32-bit.

## Status

**Production-stable in KAME** (measurement framework, 24/7 operation in
research labs since 2002).  The Phase 5 / §15–§26 work (2025 – 2026) added the
buddy chunk allocator, the full-usable page-aligned bucket tiers, the
dedicated / large-`mmap` tiers with a two-level recycle cache, pool-routed
aligned allocation, and standards-conformant OOM.

**Targets:** macOS and Windows (64-bit) for KAME itself; the standalone library
also builds and is tested on Linux (64-bit and 32-bit).  Requires a host with
`mmap` (or `VirtualAlloc`) and threads — not an MCU / bare-metal allocator.

## Benchmarks

Tight alloc/free loop, one slot at a time (`bench_multi`), median of repeated
runs.  `kame` is `LD_PRELOAD`'d against the same binary as the others; all are
default-Release builds (no `-flto` / `-march=native` — mimalloc and jemalloc
ship the same way).

**x86-64, Intel Xeon @ 2.1 GHz (cloud VM, 4 vCPU), single thread, M ops/s:**

| size    | system | mimalloc | jemalloc | **kame** |
| ------- | ------ | -------- | -------- | -------- |
| 64 B    | 160    | 196      | 176      | **251**  |
| 1 KiB   | 162    | 153      | 168      | 162      |
| 16 KiB  | 82     | 101      | 82       | **160**  |
| 64 KiB  | 75     | 103      | 8        | **113**  |
| 256 KiB | 76     | 10       | 9        | **81**   |
| 1 MiB   | 78     | 9        | 9        | **82**   |
| 4 MiB   | 75     | 9        | 9        | 46       |

**Same box, 4 threads (`mt:4`), M ops/s:**

| size    | system | mimalloc | jemalloc | **kame** |
| ------- | ------ | -------- | -------- | -------- |
| 64 B    | 610    | 435      | 493      | **633**  |
| 16 KiB  | 180    | 287      | 217      | **382**  |
| 64 KiB  | 161    | 278      | 30       | **304**  |
| 1 MiB   | 168    | 3        | 31       | **214**  |

**Apple M3 (arm64, macOS), single thread, M ops/s** — kamepoolalloc at
`820c2c5f` (§28.4), median of 5 runs (`mimalloc` 3.3 preloaded via
`DYLD_INSERT_LIBRARIES`).  Both M3 tables below were measured on this commit:

| size    | system | mimalloc | **kame** |
| ------- | ------ | -------- | -------- |
| 64 B    | 93     | 249      | **292**  |
| 1 KiB   | 76     | 213      | **261**  |
| 16 KiB  | 80     | 123      | **254**  |
| 64 KiB  | 24     | **137**  | 131      |
| 256 KiB | 25     | **136**  | 131      |
| 1 MiB   | 25     | 6.7      | **131**  |
| 8 MiB   | 43     | 5.9      | **99**   |

**Apple M3, multi-thread aggregate scaling, M ops/s (1 / 4 / 8 threads):**

| size   | system          | mimalloc            | **kame**           |
| ------ | --------------- | ------------------- | ------------------ |
| 64 KiB | 12 / 1.4 / 1.0  | 68 / 228 / **327**  | 64 / 206 / 280     |
| 1 MiB  | 12 / 1.4 / 1.0  | 3.3 / 3.4 / 2.4     | **65 / 178 / 279** |

At the **1 MiB+** chunk/large tier kame is the only allocator that *scales*
with threads: system malloc's large path is lock-serialised — it runs
*backwards* (12 → 1 M ops/s as threads rise) — and mimalloc's large tier stalls
(~3 M), while kame climbs to **279 M ops/s at 1 MiB / 8 threads**.  At 64 KiB
both kame and mimalloc scale well (mimalloc marginally ahead, 327 vs 280).

> The tier-attribution stats counters (§28.2) are sharded per-thread (§28.4) so
> this MT path stays contention-free; an unsharded build collapses the large
> tier to ~13 M ops/s at 8 threads.

**Ohtaka (ISSP supercomputer — AMD EPYC, 128-core / 8-NUMA-node, Linux 4 KiB
pages, `THP=always`), `srun --exclusive`, single-binary self-validation via
`tests/alloc_tune_report`:**

Aggregate M ops/s (`alloc → touch byte 0 → free` loop) at 1 / 4 / 16 / 64 /
128 concurrent threads:

| size,tier         |  1T  |  4T  |  16T |  64T | 128T |
| ----------------- | ---: | ---: | ---: | ---: | ---: |
| 64 B  (bucket)    |  120 |  479 | 1914 | 6459 | **11094** |
| 1 KiB (bucket)    |   69 |  284 | 1126 | 3807 |  **6174** |
| 64 KiB (chunk)    |   40 |  156 |  630 | **2213** |  1833 |
| 1 MiB (chunk)     |   44 |  173 |  691 | **1077** |   624 |
| 8 MiB (large_va)  |   25 |   44 |   63 |   70 |    99 |

The bucket tier reaches **11 G ops/s aggregate for 64 B at 128 cores
(92× linear)** — vindicating the per-thread DLL + per-thread freelist +
per-thread L1 design on a heavily-NUMA host.  The chunk tier peaks at
**2.2 G ops/s for 64 KiB at 64 cores (55×)** and **1.1 G at 1 MiB**: the
per-thread L1 with sharded ($§28.4$) tier-attribution counters absorbs the
churn without inter-core contention.  Both tiers slightly soften at 128T as
cache lines start to bounce across all 8 NUMA domains — saturating, not
regressing.  The large_va tier (8 MiB+) plateaus because the benchmark
touches byte 0 each cycle, which serialises across cores in the Linux
process-wide `mmap_lock` page-fault path; real KAME workloads write the
whole buffer right after alloc (`memcpy` of acquisition data, FFT in-place)
so the one-page fault is amortised inside the buffer write, not paid
per op.

The same `alloc_tune_report` run self-validated the defaults on this hardware:

- **`LRC_LAZY_INTERVAL_NS = 10 ms`**: 2.6 % per-thread wallclock pressure —
  printed "default 10 ms is fine; auto-tune kept it (raise-only)".
- **`LRC_K_MAX = 256`**: matched to 128 cores — printed "default is
  appropriate".
- **`MADV_HUGEPAGE`**: ineffective on this kernel (same `µs/page` as plain
  first-touch) because `THP=always` is already auto-promoting 4 KiB pages to
  2 MiB at the kernel level.
- **TLB shootdown**: ~21× worst-case at 128 threads (saturates at ~1.5 ms per
  `munmap`) — bounded, not catastrophic; the warm cache absorbs most syscalls
  in practice.

kame leads decisively in the **16 KiB – 4 MiB** range — where mimalloc and
jemalloc fall back to per-call `mmap`/`munmap` (≈ 8–10 M ops/s) while the
recycle cache keeps kame at memory-warm speed — and at multi-thread large
sizes.  In the tiny-object hot loop (≤ 1 KiB) it matches mimalloc; jemalloc's
tcache edges ahead on some sizes.  These are **no-touch** micro-benchmarks —
for workloads that *touch* the buffers the per-op cost is dwarfed by memory
bandwidth and the allocators converge.

The point is not the micro-benchmark peak but the **flat curve**: kame has no
size cliff and no per-thread working-set cliff, so a real mixed workload (small
objects + large arrays/waveforms — KAME's own profile) never hits the
`mmap`-per-large-alloc wall the others do.

## Build

### qmake (KAME-integrated)

```bash
cd kamepoolalloc
qmake kamepoolalloc.pro
make
# produces libkamepoolalloc.dylib (macOS) / libkamepoolalloc.so (Linux)
```

### Standalone (cmake test scaffold)

```bash
cd tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
ctest --output-on-failure
```

`ctest` runs the C-API conformance test.  The repo also ships manual
correctness / perf drivers built alongside it: `alloc_stress_test`
(adversarial multi-thread, sentinel-checked), `alloc_minimal_bench`
(single-size hot / fifo loops), and `alloc_bucket34_repro`.  Sanitizer
coverage (TSAN / UBSAN / ASan) is obtained by configuring the build with
the matching `-fsanitize=` flags.

## Usage

### As a global `new` / `delete` replacement

```cpp
#include "allocator.h"

int main() {
    KamePooledAllocGuard guard;  // RAII: activates the pool for this process
    auto *p = new char[256];     // routed to the pool's 256-B bucket
    delete[] p;                  // back to the per-thread freelist
}
```

Pre-`main()` allocations (dyld init, static ctors) stay on libsystem
malloc.  The guard MUST be the first statement in `main()` for the pool
to be active throughout the program.

### Runtime memory cap

```cpp
#include "allocator.h"

int main() {
    KamePooledAllocGuard guard;
    // Cap at 128 MiB: pool will not mmap more than 4 regions
    // (32 MiB each, rounded up).  Beyond the cap, allocations
    // fall back to libsystem std::malloc.
    kame_pool_set_max_bytes(128 * 1024 * 1024);

    // ... your code ...

    fprintf(stderr, "pool reserved: %zu KiB\n",
            kame_pool_reserved_bytes() / 1024);
}
```

Pass `0` to disable the cap (default = compile-time ceiling: 100 GiB
on 64-bit, 3 GiB on 32-bit).

### Explicit C API (no C++ required)

When you cannot interpose `new`/`delete` (static link, sandbox, FFI),
include `<kame_pool.h>` and call the pool directly.  Pure C linkage,
fully reentrant; pre-activation calls transparently fall through to libc.

```c
#include <kame_pool.h>

void *p = kame_pool_malloc(64);
void *q = kame_pool_aligned_alloc(4096, 1 << 20);   // 1 MiB, page-aligned
size_t cap = kame_pool_malloc_usable_size(p);        // bucket-rounded size
kame_pool_free(q);
kame_pool_free(p);

kame_pool_stats_t st = { .version = KAME_POOL_STATS_VERSION };
kame_pool_get_stats(&st);   // regions / live chunks / claimed units
```

### C API reference

Quick index of every public symbol in `<kame_pool.h>`.  All are
pure C linkage, noexcept (`KAMEPOOLALLOC_NOEXCEPT`), thread-safe, and
transparently fall through to libc on pre-activation / post-teardown.
See the header for full per-function semantics.

**Allocation (libc-equivalent surface):**

| Symbol | Purpose |
|---|---|
| `void *kame_pool_malloc(size_t)` | malloc; routes to the bucket/chunk/large-va/huge tier by size |
| `void *kame_pool_calloc(size_t n, size_t sz)` | calloc; mmap'd tiers come zeroed for free |
| `void *kame_pool_realloc(void *, size_t)` | realloc; resize-in-place when the bucket fits, else move |
| `void  kame_pool_free(void *)` | free; libc-foreign pointers transparently pass through |
| `void *kame_pool_aligned_alloc(size_t align, size_t sz)` | C11 aligned_alloc |
| `int   kame_pool_posix_memalign(void **, size_t align, size_t sz)` | POSIX-style aligned malloc |
| `size_t kame_pool_malloc_usable_size(const void *)` | bucket-rounded usable size |

**Runtime caps:**

| Symbol | Default | Purpose |
|---|---:|---|
| `void   kame_pool_set_max_bytes(size_t)` | 100 GiB / 3 GiB 32-bit | upper bound on `reserved` (region VA) |
| `size_t kame_pool_get_max_bytes(void)` | — | current cap |
| `size_t kame_pool_reserved_bytes(void)` | — | bytes of 32-MiB regions currently mapped |
| `void   kame_pool_set_large_cache_cap(size_t)` | ≈ 2 GiB total | LRC_MMAP/CHUNK recycle-cache total cap |
| `size_t kame_pool_get_large_cache_cap(void)` | — | current cap |

**Background maintenance (silenceable for realtime work):**

| Symbol | Default | Purpose |
|---|---|---|
| `void   kame_pool_set_lazy_drain_interval_ms(unsigned)` | 10 ms (auto-tuned at startup) | §28.1 lazy drain interval — bigger = fewer munmap ticks |
| `unsigned kame_pool_get_lazy_drain_interval_ms(void)` | — | current interval |
| `void   kame_pool_set_thread_exit_reclaim(int)` | on | §21 madvise(MADV_DONTNEED) at worker exit |
| `void   kame_pool_set_realtime_mode(int)` | off | §30 one-shot preset: silences all three of the above |

**Observability:**

| Symbol | Purpose |
|---|---|
| `void   kame_pool_get_stats(kame_pool_stats_t *)` | snapshot of regions/units/chunks/cache/tier counters; versioned struct (`KAME_POOL_STATS_VERSION`) |

## Tuning

Most consumers don't need to touch these — the defaults are picked for a few-
to a few-hundred-core machine.  Override via `-D…` at compile time.

### `LRC_K_MAX` — slot count per (idx, kind) in the global L2 recycle cache

Each `LrcKArray` is one cache line (or more) on its own — pushes from
different threads to the same idx land on different cache lines as long as
their `kame_owner_id() & (LRC_K_MAX − 1)` start positions differ.  K therefore
sets the **upper bound on concurrent pushers to one size class that won't
collide on a cache line**.

The default `LRC_K_MAX = 256` is the conservative choice for many-core NUMA
servers.  Smaller K reduces the static slot-array memory AND the cold pop scan
length (each pop scans up to K cache lines for a fit); larger K spreads
contention further.  K must be a power of two.

| Use case | LRC_K_MAX | Static memory (global L2) | Pop cold scan worst |
|---|---:|---:|---:|
| Desktop / few cores | 32 | 10 KiB | 32 lines |
| Single-socket server (~64 cores) | 64 | 20 KiB | 64 lines |
| Multi-socket NUMA (~256 cores) — default | 256 | 80 KiB | 256 lines |
| Huge NUMA (≥ 512 cores or many domains) | 512 | 160 KiB | 512 lines |

K-major (current) over N-major: N-major would compact a band into one cache
line (1-line pop scan) but force every concurrent pusher of that size onto a
SHARED line — catastrophic cache-line bouncing across NUMA nodes.  K-major's
per-K cache-line independence keeps inter-NUMA coherence traffic minimal,
which is the dominant cost on a 256-core node.  Trade-off: pop cold scan is
~K cache-line loads (≈ K × ~50 ns NUMA-remote / ~5 ns hot-cache).

### `LRC_N_MAX` — top size class in the recycle cache

The cache covers `[LRC_LO, LRC_HI]` at 4 indices per octave (= ~19% per
step).  `LRC_LO = ALLOC_MIN_CHUNK_SIZE = 256 KiB`; `LRC_HI = LRC_LO <<
(LRC_N_MAX / 4)`.  Default `LRC_N_MAX = 32` ⇒ `LRC_HI = 64 MiB`.  Sizes above
`LRC_HI` bypass the cache (the index space would otherwise saturate at the
top slot and over-satisfy smaller huge requests — see §27).  Must be a
multiple of 4.

| LRC_N_MAX | LRC_HI | Use case |
|---:|---:|---|
| 24 | 16 MiB | Constrained RSS; never reuse > 16 MiB |
| 32 (default) | 64 MiB | General — covers typical "large buffer" sizes |
| 40 | 256 MiB | Image / FFT / NN workloads with large buffer reuse |
| 48 | 1 GiB | Massive buffer reuse (HPC) |

Raising LRC_N_MAX also adds 4 more idx slots × LRC_K_MAX (`sizeof(atomic) ×
roundup(...)`) per octave to the static slot array.

### `LRC_K_L1` / `LRC_N_MAX_L1` — per-thread L1

L1 is per-thread (TLS), no atomics, no false-sharing concern.  `LRC_K_L1 = 32`
(fixed), `LRC_N_MAX_L1 = 24` covers idx up to 16 MiB.  Per-thread footprint
= `LRC_K_L1 × (LRC_N_MAX_L1 + 1) × 8 B ≈ 6.4 KiB`.  Raise for workloads with
many distinct per-thread sizes.

### `LRC_LAZY_INTERVAL_NS` — §28.1 amortised drain interval

Hardcoded at 10 ms.  Each LRC_MMAP push past this interval evicts one slot
to keep the steady-state cache from growing unboundedly.  On platforms where
`munmap()` is expensive (e.g. very many-core systems with TLB-shootdown
cost), raising this to 100 ms (or higher) trades drain rate for fewer
`munmap` syscalls.  Not currently a `-D…` knob; trivial to expose if
measurement shows it matters.

### Auto-characterise the host (`alloc_tune_report`)

The cmake test suite includes a runnable diagnostic that measures the
host's `mmap` / `munmap` / `madvise` costs, sweeps multi-thread munmap
(to expose TLB-shootdown scaling), benchmarks kamepoolalloc throughput at
every tier, and emits a recommendation block.  Build + run:

```sh
cmake --build <build-dir> --target alloc_tune_report
./<build-dir>/alloc_tune_report [seconds-per-bench, default 2]
```

It is NOT registered as a ctest (long-running, machine-specific output);
intended to be run once per target and the output captured for tuning
discussions.  Recommends concrete `-D…` rebuild flags when it detects
that the defaults are inappropriate (e.g. munmap so expensive that
`LRC_LAZY_INTERVAL_NS` should be raised, or `LRC_K_MAX` mis-sized for the
host's core count).

### Notes for many-core NUMA targets

- TLB shootdown on `munmap` / `madvise(MADV_DONTNEED)` scales with core count.
  The warm cache absorbs most of these — push/pop hit avoids the syscall.
- Pool regions (the 32 MiB units holding bucket-tier + dedicated-chunk
  allocations) are mmap'd push-only — once warmed up there is no munmap on
  them.  The munmap cost is paid only by the §19/§27 large-tier path on cache
  miss/eviction.
- `madvise(MADV_HUGEPAGE)` is NOT currently requested.  If THP is enabled on
  the system, the kernel may transparently coalesce 2 MiB huge pages anyway;
  explicit `MADV_HUGEPAGE` could reduce page-table footprint and TLB-shootdown
  cost.  Not yet measured.

## License

**Dual-licensed under your choice of EITHER:**

- **Apache License, Version 2.0** — see [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
  Best for embedding into permissive / proprietary projects.
- **GNU GPL, version 2 of the License, or (at your option) any later version**
  — see [LICENSE-GPL-2.0](LICENSE-GPL-2.0).
  Best for linking into GPLv2-only projects such as KAME itself.

Pick whichever license suits your downstream project; see [LICENSE](LICENSE)
for the full dual-grant statement.

Copyright (C) 2002-2026 Kentaro Kitagawa &lt;kitag@issp.u-tokyo.ac.jp&gt;,
The University of Tokyo, ISSP.

Both license grants explicitly require preservation of the copyright notice
and the choice-of-license clause when redistributing this software, in source
or binary form.

## Roadmap

See [`design/`](design/) for the structural invariant catalogue
([`INVARIANTS.md`](design/INVARIANTS.md) — the blast-radius map for safe
edits) and the §-chapter → subsystem → code navigation map
([`SUBSYSTEMS.md`](design/SUBSYSTEMS.md)).  For deeper rationale see
`git log kamepoolalloc/allocator.cpp` and the §-numbered design notes
([`CHUNK_CLAIM_TLA_NOTES.md`](tests/CHUNK_CLAIM_TLA_NOTES.md),
[`LARGE_RECYCLE_DESIGN.md`](LARGE_RECYCLE_DESIGN.md)).

### Done

- [x] Runtime memory caps (`kame_pool_set_max_bytes` region ceiling +
      `kame_pool_set_large_cache_cap` recycle-cache footprint)
- [x] Apache-2.0 relicense
- [x] Explicit C API (`kame_pool_malloc` / `_free` / `_calloc` / `_realloc` /
      `_aligned_alloc` / `_posix_memalign` family) — `<kame_pool.h>`
- [x] `kame_pool_malloc_usable_size()` public
- [x] Statistics API (`kame_pool_get_stats` — regions, live chunks, units)
- [x] Aligned-alloc served from the pool (§17)
- [x] User-installable OOM handler (`std::new_handler` loop + `bad_alloc`) (§18)
- [x] Large-tier `munmap` on free — VA returned to the OS (§19)
- [x] TSAN / UBSAN (incl. `vptr`) / ASan clean; chunk-claim/recycle TLA+
      model-checked; large-recycle cache ownership/release GenMC (RC11)-checked
      ([`tests/cds/cds_lrc_ownership.c`](tests/cds/cds_lrc_ownership.c))
- [x] 32-bit verified; 16 KiB-page (Apple Silicon) / 64 KiB-page (POWER)
      page-multiple slot layout (§16)

### Future / nice-to-have

- [ ] Static-buffer mode (`kame_pool_init(buf, len)`) — no `mmap` dependency
      (toward MCU / bare-metal use)
- [ ] Multiple heap instances (per-subsystem isolation)
- [ ] Unified `KAME_POOL_*` env / config surface (hugepages, prewarm, caps)
- [ ] `fork()` safety (`malloc_disable` / `_enable`)
- [ ] musl / uclibc, RISC-V verification

## Design timeline

Carved out of KAME and generalised from "KAME-specific small-object pool" to a
four-tier general allocator.  Selected milestones (full history in `git log`):

| §     | subject |
| ----- | ------- |
| 5l–5u | buddy 32 MiB regions, multi-unit chunks, bucket layout, runtime cap, Apache-2.0 |
| 13    | O(1) `p → chunk` via 2-level radix tree (retires the O(N) region scan) |
| 13.2/13.3 | per-region metadata in unit 0; push-only region list (retire cap-sized globals) |
| 14    | NUMA-aware claim; opt-in hugepages; `kame_pool_get_stats` |
| 15    | forward-shift — every chunk's slot region starts on a 256 KiB boundary |
| 16    | full-usable ALIGN ≥ 1024 tiers — 0 % round-up at power-of-2 page sizes |
| 17    | pool-routed `posix_memalign` / `aligned_alloc` / `operator new(align_val_t)` |
| 18    | standards-conformant OOM (`new_handler` + `bad_alloc`; nothrow / errno) |
| 19    | large-`mmap` tier (4–32 MiB) with real `munmap` on free |
| 20    | fix cross-thread-free `vptr`-after-release UB (UBSAN) |
| 21–23 | thread-exit reclaim default; recycle cache; IE-TLS hot-path slots |
| 24    | `slow_allocate` scans DLL freelists across chunks (multi-chunk working sets) |
| 25    | global lock-free log-slot recycle cache (no working-set cliff) |
| 26    | per-thread L1 recycle log + global L2 — MT scaling without the cliff |

## Acknowledgements

Designed and implemented by Kentaro Kitagawa.  Phase 4 / Phase 5 refactor
work co-authored with Anthropic Claude (Opus 4.5–4.7) under explicit
direction; all algorithmic and performance decisions reviewed and verified
by the original author.
