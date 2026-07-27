# kamepoolalloc §75 realtime work — Linux-side handoff (**CLOSED**)

Both items are done. The results live in `design/RT_READINESS.md` — §G9 for
item 1, §G6(a) for item 2 — which is the permanent home; this file is kept
for the *method*, and in particular for the environment traps below, which
cost real time to find and which anyone re-running these measurements on
Linux will hit again.

Host used: 4 vCPU Intel Xeon @ 2.80 GHz (Firecracker microVM, 16 GiB),
Ubuntu 24.04, glibc 2.39, kernel 6.18.5, GCC 13.3, Release. `ctest` 18/18
green on **both x86-64 and `-m32`** (see the ILP32 note below).

**Not a realtime kernel.** `PREEMPT_DYNAMIC`, not `PREEMPT_RT`
(`/sys/kernel/realtime` absent); no `isolcpus` / `nohz_full` / `rcu_nocbs`;
`sched_rt_runtime_us` at the default 950 ms/1 s, so the `SCHED_FIFO prio 80`
the harness reports getting is still throttled and preemptible; 4-vCPU guest,
so steal time is in every sample.  The G6(a) mechanism results (does
`MADV_NOHUGEPAGE` actually stop 2 MiB zeroing) are unaffected by that — they
are page-fault-path facts — but the `MAX` / `p99.99` cells are not WCET
numbers and §G6(a) says so explicitly.  Anyone re-running this to establish a
bound needs a `PREEMPT_RT` host with the measured thread on an isolated,
`nohz_full` core.

---

## Item 1 — G9 negative control: does the regression test have teeth? — **YES**

The question was narrow: with the `3145e139a` guard reverted, does
`alloc_thread_exit_unarmed_test` fail on Linux? If not, the test is toothless
and needed strengthening rather than closing.

It fails, with exactly the predicted signature.

| | `units_live` | `chunks_live` |
|---|---|---|
| guard present (both linkages) | 6 → 10, plateau from cycle 40 | 5 → 6, plateau |
| guard reverted (both linkages) | 26 → 129 | **25 → 125 over 100 cycles = +1/cycle** |

The revert was minimal — `|| kame_thread_torn_down()` dropped from the
`l1_push` guard and nothing else, restoring exactly the pre-`3145e139a` gap.
`l1_pop_fit`'s guard (which is about `g_lrc_l1_threads` counter drift, not
stranding) was left alone. Guard restored afterwards; nothing reverted was
committed.

Two things confirmed rather than assumed, both recorded in §G9:

* **glibc ordering.** A probe registering both a C++ `thread_local` destructor
  and a `pthread_key` destructor on one thread shows the `thread_local` one
  running first on glibc 2.39 — the window is real, and it is the window macOS
  does not open.
* **Which term is load-bearing.** Instrumenting `l1_push` at the moment the
  consumer's `pthread_key` destructor frees the foreign block shows
  `s_l1_drained == 0` but `kame_thread_torn_down() == 1`. The consumer never
  armed its L1, so the pre-existing `s_l1_drained` check cannot fire; its
  *bucket-tier* TLS was armed, so `AllocThreadExitCleanup`'s `thread_local`
  destructor had already set `s_alloc_tls_off`. The added term is the only
  thing closing the window, which is what makes the test a real regression
  test for `3145e139a` and not a coincidence.

---

## Item 2 — G6(a) `MADV_NOHUGEPAGE` — **implemented, opt-in, measured**

Shipped as `kame_pool_set_thp_policy()` / `kame_pool_get_thp_policy()` with
`KAME_THP_SYSTEM` / `ALWAYS` / `NEVER`. Full rationale, numbers and the
"should realtime mode imply it?" decision (**no**, and the measurement backs
the instinct: up to +58 % on a TLB-bound working set is too surprising for a
knob documented as silencing background maintenance) are in §G6(a).

Three things the original plan did not anticipate, all found by measuring:

1. **The large-VA tier needed the advice more than the regions did.** The plan
   named only `mmap_new_region()`. But blocks above `LRC_HI` are a fresh
   32 MiB-aligned mmap per allocation and are the coldest, largest memory the
   pool hands out — advising only regions left every one of their 2 MiB spans
   faulting as a hugepage. `large_va_raw_map()` now carries the same policy.
2. **`MADV_NOHUGEPAGE` does not split hugepages that already exist.** It stops
   future hugepage faults and future khugepaged collapses, which is what
   matters for latency, but a range already backed by THP stays backed. Hence
   the documented call order: policy **before** prewarm.
3. **`KAME_THP_SYSTEM` cannot be re-applied.** Linux has no "clear" advice;
   `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE` each clear the other's flag and
   neither restores neutral. Policy 0 returns 0 from the re-advise walk and
   affects new regions only.

### Traps to know before re-running any of this on Linux

* **Containers commonly set `PR_SET_THP_DISABLE` on the whole process tree.**
  This one invalidates everything silently: every VMA reports
  `THPeligible: 0`, `AnonHugePages` stays 0 no matter what you advise, and
  both the "policy 2 holds it at 0" check and every latency arm pass for the
  wrong reason. `/sys/kernel/mm/transparent_hugepage/enabled` says `[always]`
  the whole time and tells you nothing.
  Check `THP_enabled:` in `/proc/self/status` (0 = disabled). Clear it with a
  tiny wrapper that calls `prctl(PR_SET_THP_DISABLE, 0)` and then `execvp`s
  the real binary — the flag is in `MMF_INIT_MASK`, so the cleared state
  survives the `exec`. **Always run a control first**: a plain
  `mmap` + `MADV_HUGEPAGE` + `memset` should show `AnonHugePages` equal to the
  touched size. If the control is 0, nothing downstream means anything.
  (Sub-test (2e) in `alloc_rt_thread_test` encodes this: it measures a
  baseline and *skips* rather than asserting when the host backs no THP.)
* **`bench_rt_wcet` could not see a hugepage fault at all**, for two reasons
  worth knowing: its first touch was outside the timed window (the
  `now_ns()` pair bracketed only `kame_pool_malloc`), and even once timed, the
  steady-state loop is prewarmed and the large tier hands back a pointer whose
  first page the allocator itself has already written a header into. Fixed by
  timing the touch (`1st-touch` histograms) and adding `--faults`, which times
  one write per 4 KiB page across freshly-mapped `> LRC_HI` memory. `--thp
  system|always|never` selects the arm.
* **The arms cannot be interleaved inside one process** — see (2) above, the
  policy cannot be un-applied to faulted memory. A/B across *processes*,
  alternating order, median of ≥ 7. On a 4-vCPU shared VM the max is dominated
  by preemption (single-run maxima ranged 0.98–32.8 ms for the same arm), but
  p50/p99.9/p99.99 medians were stable to within a bucket across independent
  sessions.
* **`bench_loop` cannot measure the THP cost** — it keeps one block live, so
  its working set is TLB-trivial. `tests/bench/bench_tlb.c` was added for
  this: a dependent random pointer chase over a pool-allocated working set,
  which is deliberately the *worst* case for TLB reach.
* **Do not use an *unadvised* range as the "THP is on" baseline.** Under the
  common `defrag = madvise` setting the kernel will not compact to find a
  2 MiB block for a range nobody asked about, so a `KAME_THP_SYSTEM` baseline
  reads 0 kB as soon as memory is mildly fragmented — and a check written as
  "baseline > 0, then assert NEVER == 0" then skips itself for no reason.
  With a `SYSTEM` baseline, sub-test (2e) passed 64-bit and skipped 32-bit on
  the same host and kernel. Use `MADV_HUGEPAGE` as the baseline arm; it earns
  the compaction effort and makes the A/B deterministic.

### ILP32

Worth doing, since the standalone library claims Linux 32-bit support and
this work touches page-level code. `apt-get install gcc-multilib
g++-multilib`, then configure with `-DCMAKE_C_FLAGS=-m32
-DCMAKE_CXX_FLAGS=-m32 -DCMAKE_EXE_LINKER_FLAGS=-m32
-DCMAKE_SHARED_LINKER_FLAGS=-m32`. Result: 18/18, warning-free, and (2e)'s
behavioural half passes with the same 10,240 kB vs 0 kB figures as 64-bit —
THP is a kernel page-table property, not a function of pointer width, so an
ILP32 process on an x86-64 kernel gets hugepages normally.

The two defects that build found were both in the new test/bench code, not
the library: a `%lx`-into-`uintptr_t` scan in (2e)'s smaps parser (ILP32
`uintptr_t` is `unsigned int`), and a `size_t` overflow in `bench_tlb` for a
multi-GiB working set that surfaced as a misleading "working set too small".
Both fixed. The library itself needed nothing — its ILP32 handling
(`ALLOC_MAX_REGIONS = 96` = 3 GiB, `RADIX_VA_LIMIT = ~0`) predates this work.

### Reproducing

```bash
cd kamepoolalloc && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j && (cd build && ctest --output-on-failure)

# THP must be available AND not prctl-disabled — see the traps above.
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

./build/tests/bench_rt_wcet --faults 24 --thp system   # tail, THP arm
./build/tests/bench_rt_wcet --faults 24 --thp never    # tail, anti-THP arm
./build/tests/bench_tlb 512 1024 6000000               # TLB cost, default
KAME_POOL_NOHUGEPAGE=1 ./build/tests/bench_tlb 512 1024 6000000
```

---

## Context you may want

- `design/RT_READINESS.md` — the whole programme, G1–G10, with what is claimed
  and what explicitly is not. G6(a) and G9 carry these results.
- README §"The realtime contract" — preconditions → guarantees → exclusions.
- `tests/alloc_rt_thread_test.cpp` — asserts each RT claim by observing a
  counter move rather than trusting that a call was made; (2d) is the G6(b)
  `mlock` one and (2e) the G6(a) THP one. Both are written to tolerate a
  hostile CI environment (low `RLIMIT_MEMLOCK`; no THP) by skipping visibly
  rather than passing vacuously. This host: `ulimit -l` = 8192 kB, so `mlock`
  returns a short count — documented behaviour, not a failure.

## Two methodology traps already paid for (from the macOS side)

- **Deep-tail comparisons need ~10⁶ samples.** At 120 k the cross-thread arms
  ordered *backwards* at p99.9; at 4 M they were equal and the RT arm won the
  deep tail. Below ~10⁶ the deep-tail buckets hold single digits and the
  ordering is noise.
- **`p50 = 0 ns` is the clock floor, not a sub-nanosecond free.** Apple
  Silicon's `steady_clock` ticks at ~41.7 ns. On this x86 host the floor is
  far finer (clock-overhead mean 22 ns), which is why the 27 ns vs 2,048 ns
  p50 split in the fault measurements is readable at all.
