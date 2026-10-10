# Handoff: v1.2.0 against v1.1.0 — where it is faster, and where it is slower

Started 2026-10-10 on a MacBook Air M3.  **Open:** the regressions below are
measured on one laptop; the M5 Ultra and Ohtaka runs (§4) decide what they
are.  Append results here.

## 1. Summary

v1.2.0 removes the chunk-list walks from the allocation path, and on the
workload that walk hurt it is two orders of magnitude faster.  It is also
slower than v1.1.0 in two places nobody had measured:

- **larson** (mimalloc-bench; thread churn with cross-thread frees):
  0.67–0.74x.  In two steps, §fl-avail and §revive.
- **bench_loop at 1 KiB and 16 KiB** (single thread): 0.85–0.93x; 64 B is
  unchanged.  §fl-avail, partly recovered by §revive, and then the TSan
  relaxed atomics, which cost 0.94x on their own.

Two published statements rest on a narrower measurement than they read:

- The v1.2.0 tag message: "single-thread hot paths within +-5 %".  That is
  §revive's own number — `3b353c5e6` → `39edad5a9`, M5 Ultra,
  `bench_loop_pool` — not v1.1.0 → v1.2.0.
- The drafted mimalloc-bench pin PR: "single-thread paths are unchanged".
  If it is not posted yet, hold it until §4 is in.

## 2. How to reproduce

`tests/bench/bench_ab.sh` (`32ff4c601`) builds both tags with
mimalloc-bench's recipe, runs every workload as one binary with the arm
switched by preload, and alternates the arm order between repetitions.

    tests/bench/bench_ab.sh --threads "4 8"                     # all workloads
    tests/bench/bench_ab.sh --only "loop larson" --threads "2 4"   # the two regressions

Bisecting: build the library at intermediate commits and give them as arms
(`--tags "" --arm NAME=LIB ...`); see `tests/bench/README.md`.  The split
commits are on `standalone/kamepoolalloc`; the table in §3.2 names the KAME
commits they mirror.

## 3. MacBook Air M3

Apple M3 (4P + 4E), 24 GiB, macOS 26.7, AC power; Apple clang 17.0.0; both
libraries Release, `DYLD_INSERT_LIBRARIES`.  M ops/s, higher is better;
median (min–max).  A fanless laptop: treat single-digit percentages with
care — §3.2 carries its own noise estimate.

### 3.1 v1.1.0 against v1.2.0 (5 repetitions)

| workload | v1.1.0 | v1.2.0 | ratio |
|---|---|---|---|
| bench_loop 64 B | 687 (508–703) | 687 (649–742) | 1.00x |
| bench_loop 1 KiB | 489 (474–497) | 435 (422–440) | **0.89x** |
| bench_loop 16 KiB | 558 (515–577) | 497 (476–498) | **0.89x** |
| larson, 2 threads | 58.1 (54.5–59.7) | 38.9 (35.0–41.2) | **0.67x** |
| larson, 4 threads | 98.5 (84.1–103) | 73.3 (63.8–76.1) | **0.74x** |

`bench_xlatency`, 1 KiB blocks, 200 K live, one run each: v1.1.0 median
44 µs, p99.9 108 µs, max 322 µs, 0.022 M allocs/s; v1.2.0 median 42 ns (one
clock step), p99.9 0.79 µs, max 49 µs, 6.3 M allocs/s.

Smoke only (one repetition of 1 s; not a result): bench_xthread 2 workers
1.99x, xmalloc-test 1.04x, rptest 1.01x.

### 3.2 Bisection (3 repetitions)

| arm | KAME commit | larson, 2 threads | bench_loop 1 KiB | bench_loop 16 KiB |
|---|---|---|---|---|
| v1.1.0 | — | 55.0 (54.9–59.0) | 472 | 563 |
| §fl-avail | `3b353c5e6` | 45.1 (39.4–46.9) — **0.82x** | 384 — 0.81x | 479 — 0.85x |
| §revive | `39edad5a9` | 38.8 (38.0–40.2) — **0.71x** | 444 | 516 |
| orphan-chain serial | `1dbf9e63c` | 38.3 | 448 | 510 |
| §group | `bd7875d06` | 39.0 | 459 | 520 |
| owner-id offsets | `ee9d757b0` | 38.2 | 457 | 520 |
| TSan atomics | `d6cc5007d` | 40.2 | 435 | 481 |
| v1.2.0 | — | 39.8 | 437 | 476 |

The last two arms are **the same binary** — the commits between them are
documentation and a load that compiles identically on arm64, and the two
dylibs' sha256 match — so their spread (1–3 %) is this machine's noise.

The TSan step alone, `ee9d757b0` against `d6cc5007d`, 9 repetitions:
bench_loop 1 KiB 464 (448–466) → 436 (431–438), 16 KiB 528 (487–531) →
495 (477–496), both 0.94x; 64 B 1.00x.  `23d843e23` turned the bitmap-word
loads, the `m_sizes` store and the slot-header store/load into
`__atomic_*_n(RELAXED)`: the same instructions on arm64, but the compiler
may no longer merge, reorder or keep them in registers.  That is the
likely cost — not yet checked in the disassembly.

## 4. Still to run

- **M5 Ultra** (where §revive's numbers came from):
  `tests/bench/bench_ab.sh --threads "4 8"`; paste `summary.md` and
  `provenance.txt` here.
- **Ohtaka** (the README's x86-64 reference):

      CC=~/llvm-install/bin/clang CXX=~/llvm-install/bin/clang++ \
        ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --build-only \
        --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench
      srun -p i8cpu --time=02:00:00 --exclusive \
        ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --run-only \
        --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
        --threads "4 16 64" --xlat-cpu 0,1

Then: whether to move the mimalloc-bench pin to v1.2.0 now for the
use-after-free, or to fix these first and pin v1.2.1.
