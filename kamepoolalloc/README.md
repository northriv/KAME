# kamepoolalloc

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20(MinGW)-lightgrey)]()

A lock-free, per-thread bucketed pool allocator for C++ small-object workloads.
Designed for **multi-threaded STM-style transactional workloads** but usable as a
general-purpose drop-in `new` / `delete` replacement on macOS, Linux, and
Windows (MinGW).

Carved out of the [KAME](https://github.com/northriv/KAME) measurement framework
and **dual-licensed under Apache 2.0 OR GPL-2.0-or-later** at your choice — so
it can be embedded into permissive / proprietary projects (Apache 2.0 path) or
linked into GPLv2-only projects such as KAME itself (GPL path).

## Highlights

- **Lock-free fast path** — TLS freelist pop/push, no atomics on the hot path.
- **Sized buckets to 16 KiB** — 47 size classes covering 1 B .. 17400 B with
  ≤ 12 % internal fragmentation at every power-of-2 round number.
- **Per-thread DLL chunks** — no global allocator lock, no contention until
  the chunk-claim slow path.
- **Cross-thread frees** via per-thread holding batch + bit-clear coalescing.
- **Multi-unit buddy chunks** — 1 / 2 / 4 × 256 KiB units depending on size
  class. O(1) chunk-base lookup via back-offset table.
- **Bounded VA** — runtime cap via `kame_pool_set_max_bytes()`; compile-time
  ceiling 100 GiB on 64-bit (3 GiB on 32-bit).
- **mprotect-free reclaim** — `madvise(MADV_FREE)` on macOS, `MADV_DONTNEED`
  on Linux.  No syscall on the chunk-claim / release path other than the
  initial 32-MiB-region mmap (one syscall per ~1000 small allocations).
- **Strict aliasing & C++17 clean** — no UB casts, no reinterpret-cast of
  storage-classified pointers.  Compiles under `-Wall -Wextra` clean.
- **DCAS-free** — `uint32_t` packed counter + state bits; `compare_exchange`
  on `uint64_t` only where the host guarantees `ATOMIC_LLONG_LOCK_FREE == 2`.

## Status

**Production-stable in KAME** (measurement framework, 24/7 operation in
research labs since 2002).  Phase 5 family (Aug 2025 – May 2026) added the
buddy chunk allocator, multi-tier ALIGN buckets, and the cross-thread cursor
fix that closes the last reuse-heavy workload gap.

**Embedded readiness:** WIP.  Static-buffer mode and explicit C API are
planned (see [Roadmap](#roadmap)).  Currently requires a POSIX-ish host
with `mmap` and `pthread`.

## Benchmarks (Apple M3, 4P+4E, Phase 5t)

Alloc + free in a tight loop, one slot at a time (single thread, warm cache):

| size      | rate           | vs glibc | vs mimalloc |
| --------- | -------------- | -------- | ----------- |
| 96 B      | 460 M ops/s    | 3.1×     | 2.5×        |
| 272 B     | 485 M ops/s    | —        | —           |
| 1024 B    | 390 M ops/s    | 2.4×     | 2.7×        |
| 8192 B    | 390 M ops/s    | 4.6×     | 3.7×        |

Adversarial multi-thread (`alloc_stress 200 × 32 × 100K @ 50 % cross-thread`):

| metric                  | KAME    | glibc   | mimalloc |
| ----------------------- | ------- | ------- | -------- |
| throughput              | 18.7    | 11.8    | 12.3     |
| RSS                     | 9 MiB   | ~200    | ~180     |
| peak VmSize             | 3 GiB   | 2.4     | 2.5      |

Linux numbers are similar in throughput; RSS savings depend on workload.

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

`ctest` runs the STM transaction, atomic-shared-ptr, and concurrency
verification suite that doubles as the allocator's correctness coverage
(9 tests, all must pass).

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

Listed in implementation order; see Phase commits in `git log` for design
rationale on completed items.

### Embedded readiness (in progress)

- [x] Runtime memory cap (`kame_pool_set_max_bytes`) — Phase 5u
- [x] Apache-2.0 relicense — Phase 5u
- [ ] Explicit C API (`kame_malloc` / `kame_free` family) for non-C++ consumers
- [ ] Static-buffer mode (`kame_pool_init(buf, len)`) — no `mmap` dependency
- [ ] `malloc_usable_size()` public for `std::vector` etc. compatibility
- [ ] Statistics API (`mallinfo`-style: per-bucket usage, fragmentation)

### Diagnostic / debugging

- [ ] ASAN / Valgrind transparent shim
- [ ] User-installable OOM handler
- [ ] TSAN clean (concurrent allocator is its own test case)

### Future / nice-to-have

- [ ] Multiple heap instances (per-subsystem isolation)
- [ ] Aligned-alloc pool support (currently falls through to `posix_memalign`)
- [ ] 32-bit ARMv7 / RISC-V verification
- [ ] musl / uclibc verification
- [ ] 64 KiB-page ARM64 (some Linux distros, ARM Mac under Asahi)

## Phase 5 timeline (May 2026)

Major refactor cycle that brought the allocator from "KAME-specific" to
"general-purpose small-object pool":

| Phase | Commit prefix | Subject |
| ----- | ------------- | ------- |
| 5l    | buddy chunks  | uniform 32 MiB regions, multi-unit chunks 1/2/4 |
| 5m    | region bitmap | s_region_has_free skip-bitmap for O(1) chunk claim |
| 5n    | DLL cursor    | per-thread DLL walk cursor + exhausted flag |
| 5p+5q | bucket layout | N+1 shift for round-number frag + FS=true 16-step extension to 368 B |
| 5r    | cursor reset  | reset cursor after direct batch_return_to_bitmap (Linux bucket34_repro 33.5 → 0.24 M/s root fix) |
| 5t    | TID-aware     | gate cursor reset by chunk-owner identification |
| 5u    | embedded API  | runtime memory cap + Apache-2.0 relicense |

See `git log kamepoolalloc/allocator.cpp` for full history.

## Acknowledgements

Designed and implemented by Kentaro Kitagawa.  Phase 4 / Phase 5 refactor
work co-authored with Anthropic Claude (Opus 4.5–4.7) under explicit
direction; all algorithmic and performance decisions reviewed and verified
by the original author.
