# kamepoolalloc §75 realtime work — Linux-side handoff

Two items remain from the RT-readiness programme (`design/RT_READINESS.md`) and
both need a **Linux host**; everything else is done and measured on macOS.
Self-contained: you should not need the originating session's context.

Repo conventions: work on **KAME `master`** (the standalone
`northriv/kamepoolalloc` repo is a read-only subtree mirror — never commit
there). The git remote is named **`GitHub`**, not `origin`. Pull before push,
never force-push. Commit trailer:

```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

Build + test:

```bash
cd kamepoolalloc && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cd build && ctest --output-on-failure          # 18 tests, all should pass
```

---

## Item 1 — G9 negative control: does the regression test have teeth?

**Status: the fix and the test both already exist. What is missing is proof the
test can fail.**

- Fix: `3145e139a` gates both L1 recycle entry points on
  `kame_thread_torn_down()` — `l1_push` (`allocator.cpp:6803`, beside the
  pre-existing `s_l1_drained` check) and `l1_pop_fit` (`:6770`).
- Test: `tests/alloc_thread_exit_unarmed_test.cpp`, built both statically and
  against the dylib (`alloc_thread_exit_unarmed_test{,_dynamic}`), both in
  ctest, both passing.

The problem: **macOS cannot trigger the bug at all.** glibc runs C++
`thread_local` destructors *before* `pthread_key` destructors, which is what
opens the window; dyld's ordering does not. So the test passing on macOS says
nothing about whether it actually covers the fix.

### What to do

1. Run `ctest` on Linux as-is — both linkages should pass. (If they *fail*,
   stop: that is a live bug, not a test problem. Capture the
   `chunks_live` / `units_live` progression it prints.)
2. **Revert the guard and confirm the test fails.** The minimal revert is to
   drop `|| kame_thread_torn_down()` from the `l1_push` guard in
   `allocator.cpp` (around `:6803`, the line reading
   `if(__builtin_expect(s_l1_drained || kame_thread_torn_down(), 0)) return false;`)
   — that alone restores the exact pre-`3145e139a` narrow gap. Rebuild, run
   `alloc_thread_exit_unarmed_test` and `_dynamic`.
   - **Expected:** monotonic growth, ~ +1 chunk per cycle, and a FAIL.
   - **If it still passes:** the test does not cover the fix. Most likely cause
     is that the consumer thread's `pthread_key` destructor is not running
     after the C++ `thread_local` dtors in your libc, or the consumer is
     inadvertently arming its L1. Strengthen the test rather than declaring the
     item done, and record what you found.
3. Restore the guard (`git checkout -- kamepoolalloc/allocator.cpp`) and record
   the outcome in `design/RT_READINESS.md` §G9 — replacing the "negative
   control owed" paragraph with the result, either way.

Do **not** commit the reverted state.

---

## Item 2 — G6(a) `MADV_NOHUGEPAGE` for the arena

**Why:** we ship the *pro*-THP knob and not the anti-THP one, which is
asymmetric now that the README claims realtime support. Transparent hugepages
hurt realtime three ways:

1. a first touch inside a 2 MiB-aligned range can make the kernel allocate
   **and zero a whole 2 MiB page** instead of 4 KiB — a single fault costing
   orders of magnitude more;
2. `khugepaged` may run **memory compaction** to find a contiguous 2 MiB block,
   stalling an unrelated thread's fault for milliseconds;
3. prewarming does not protect you — khugepaged can collapse the range
   afterwards, and the collapse itself takes the page-table lock.

jemalloc offers `opt.thp = never` for essentially these reasons (plus RSS
bloat); tcmalloc went the other way and manages hugepages deliberately
(Temeraire, OSDI'21). For realtime, jemalloc's side is the right one.

### Current state

`mmap_new_region()` in `allocator.cpp` (search `KAME_POOL_HUGEPAGE`, ~`:6560`)
has an opt-in block:

```cpp
#  if defined(__linux__) && defined(MADV_HUGEPAGE)
        static const bool hp = /* getenv("KAME_POOL_HUGEPAGE") */;
        if(hp) madvise(p + ALLOC_PAGE_SIZE, mmap_size - ALLOC_PAGE_SIZE,
                       MADV_HUGEPAGE);
#  endif
```

i.e. per-region, at creation, env-gated, default off.

### Suggested shape (not prescriptive — your call after measuring)

A bare `KAME_POOL_NOHUGEPAGE=1` mirror is the two-line version, but it has a
real gap: **regions are created lazily**, so a region that already exists when
realtime mode is enabled never receives the advice. Prefer a runtime policy that
also re-advises existing regions:

```c
/* 0 = leave to the system (default), 1 = MADV_HUGEPAGE, 2 = MADV_NOHUGEPAGE */
void kame_pool_set_thp_policy(int policy);
int  kame_pool_get_thp_policy(void);
```

- Store the policy in a global; apply it in `mmap_new_region()` for future
  regions (replacing the env-only path, keeping `KAME_POOL_HUGEPAGE=1` working
  as an initial value for compatibility).
- Apply it to **existing** regions by walking the per-NUMA region lists — the
  walk already exists, copy `PoolAllocatorBase::mlock_regions()` in
  `allocator.cpp` (added for G6(b)); it is the same loop over
  `s_region_dll_heads[node]` / `dll_next`, and `region_meta()` is a plain cast
  so `rm` *is* the region base. Skip page 0 (the metadata page) exactly as the
  `MADV_HUGEPAGE` block does.
- Decide whether `kame_pool_set_realtime_mode(1)` should imply policy 2. My
  instinct is **no** — silently costing TLB performance on a knob documented as
  "silences background maintenance" would be surprising — but measure first and
  document whichever you choose. If you do wire it up, mention it in the
  README's realtime contract.
- Non-Linux: no-op (macOS has no THP; keep the stub in `allocator.h` for
  `USE_STD_ALLOCATOR` builds consistent with the other `kame_pool_rt_*` stubs).

### How to verify on Linux (please do all three)

1. **The advice takes effect.** With
   `/sys/kernel/mm/transparent_hugepage/enabled` = `[always]`, allocate enough
   to create a few regions, touch them, and read `AnonHugePages:` for the
   region ranges in `/proc/self/smaps`. Policy 2 should hold it at 0 kB;
   default/policy 1 should show it growing. This is the check that the call is
   actually doing something — do not skip it in favour of the latency numbers
   alone.
2. **It removes the fault spike.** `tests/bench/bench_rt_wcet.cpp` already times
   a first touch after each `malloc` (`*static_cast<char*>(p) = k`). Run it
   under `THP=always` with policy default vs policy 2 and compare the **max /
   p99.99** of the malloc histograms, not the means. Interleave the two arms
   (the harness's own A/B does this per repetition — extend it, or run
   alternating processes) because machine state drifts. Expect the huge-page
   zeroing spike to disappear from policy 2.
3. **The throughput cost.** Run `tests/bench/bench_loop_pool` and the
   mimalloc-bench comparison you normally use, policy 2 vs default, on a
   TLB-bound (large working set) case. If the cost is large, that is an argument
   for keeping it opt-in — record the number either way, since the README will
   need to state the trade.

Then update: `design/RT_READINESS.md` §G6(a) (status + numbers), the README
realtime-contract exclusion that currently says THP "remains a hazard we do not
yet gate", and the API table.

---

## Context you may want

- `design/RT_READINESS.md` — the whole programme, G1–G10, with what is claimed
  and what explicitly is not.
- README §"The realtime contract" — preconditions → guarantees → exclusions.
- `tests/alloc_rt_thread_test.cpp` — asserts each RT claim by observing a
  counter move rather than trusting that a call was made; sub-test (2d) is the
  G6(b) `mlock` one and is written to tolerate a low `RLIMIT_MEMLOCK`, which
  CI containers often impose. Worth checking what your host's limit is
  (`ulimit -l`) — if it is small, `kame_pool_mlock_regions()` will return a
  short count, and that is the documented behaviour, not a failure.
- Measured on macOS M3 for reference (yours will differ): free on the
  > 256 MiB band, realtime vs default — median 128 ns vs 20,480 ns, max 792 ns
  vs 677,917 ns; 32 B cross-thread free max 27 µs vs 107 µs.

## Two methodology traps already paid for

- **Deep-tail comparisons need ~10⁶ samples.** At 120 k the cross-thread arms
  ordered *backwards* at p99.9; at 4 M they were equal and the RT arm won the
  deep tail. Below ~10⁶ the deep-tail buckets hold single digits and the
  ordering is noise.
- **`p50 = 0 ns` is the clock floor, not a sub-nanosecond free.** Apple
  Silicon's `steady_clock` ticks at ~41.7 ns. Check your host's granularity
  before reading anything into sub-100 ns figures; on x86 with a TSC-backed
  clock you will have far finer resolution, which is an advantage worth using —
  the macOS numbers above are floor-limited in the small bands and yours need
  not be.
