# kamepoolalloc — RT readiness: current guarantees and remaining work

Audit of what the allocator already guarantees for real-time (RT) use and
what remains before a **bounded per-op WCET contract** can be stated and
defended.  Context: making the KAME STM hard-RT requires the allocator
underneath it to be RT first — every STM commit clones Payloads through
this pool.

Method: grep-audit of `allocator.cpp` / `allocator_prv.h` (2026-07-26,
master) for syscalls, unbounded loops, and blocking on the alloc/free
paths, cross-checked against the §-tagged design notes.  Line numbers are
of that snapshot; prefer the §-tags when they drift.

---

## 1. What already holds (evidence)

| Property | Evidence |
|---|---|
| **No mutexes, no sleeps, no yields** anywhere in the allocator | grep for `pthread_mutex\|std::mutex\|futex\|usleep\|nanosleep\|sched_yield\|this_thread` — zero hits outside comments. All synchronisation is CAS / atomics / seqlock. |
| **Hot path is short and TLS-local** | word-cache (default ON): 1 CAS steal → TLS word → 1 `ctz` per alloc. FS=false allocate is walk-once-bail, bounded by `m_count` words (`allocator.cpp:1580` comment). |
| **§30 realtime mode exists** (`kame_pool_set_realtime_mode(1)`, `allocator.cpp:7184`) | Silences the three *background* maintenance paths: (1) §28.1 lazy-drain tick (the only munmap outside an explicit alloc/free), (2) §28.3 auto-tune munmap probe, (3) §21 thread-exit madvise (`s_thread_exit_reclaim`). |
| **Steady-state 32 KiB–32 MiB band is syscall-free** | §22/§25/§28 two-tier LRC recycle cache + global log-slot cache: hits recycle VA without mmap/munmap. |
| **Chunks park warm on release** | §34 `bucket_release_chunk` parks a fully-freed bucket chunk for re-claim instead of unmapping — re-acquire is a bitmap CAS, not an mmap. |
| **Pre-warm idiom documented** | `contrib/README.md` "When NOT to use these on a hard-real-time path" + `contrib/ros2_allocator.hpp`: TLSF-style allocate+free of the working-set sizes before entering the RT loop. |
| **Memory cap API** | `kame_pool_set_memory_cap` (`allocator.h:168`) — refuses fresh regions past the cap (upper bound, not a pre-reserve). |
| **Single pool-region mmap + bitmap-claim site** (§74, done — `c04a7975d`) | `allocate_chunk<ALLOC>()` is LRC-pop → `claim_chunk` → header-stamp only; `mmap_new_region()` has exactly one caller, `claim_chunk` (`allocator.cpp:3957`). Both the bucket templates and the dedicated large path share it. |
| **Protocol correctness machine-checked** | chunk-claim / recycle / orphan-chain protocols TLA+ + GenMC (RC11) verified (see `tests/`); this doc is about *latency bounds*, not correctness. |

### 1.1 The RT audit surface: every mapping site

Because of §74 the whole allocator has only four places that can enter the
kernel for memory, which is what makes the G1–G3 gating tractable:

| Site | Tier | Reached from | RT status |
|---|---|---|---|
| `mmap_new_region()` (`:6360`) ← `claim_chunk` (`:3957`) | pool regions (32 MiB) | cold chunk claim (bucket + dedicated) | unbounded by nature → must be pushed out of the RT section by prewarm (G2) / pre-reserve (G3) |
| `radix_alloc_l2()` (`:6277`) | radix L2 leaf (8 KiB) | first insert into an uncovered VA band | same — prewarm must touch the radix path (G2.1) |
| `large_va_raw_map()` (`:6507`) ← §19 large tier (`:7258`) | > dedicated, ≤ 32 MiB | large alloc missing the LRC | LRC hit avoids it; free-side `munmap` is G1 |
| `lrc_auto_tune_lazy_interval()` (`:6936`) | one-shot probe | first LRC_MMAP push | **already gated** by §30 (`g_lazy_auto_tune_done`) |

Plus the page-reclaim syscalls (`madvise`) in `deallocate_chunk` (`:3264/:3273`)
— the G1 hole.

## 2. Gaps (the remaining work), prioritized

### G1 — free-path reclaim on a realtime thread — **DONE**

Implemented as a **per-thread** gate (`kame_pool_set_realtime_thread`, or the
`kame::rt_section` RAII guard), because the reclaim syscall runs on the
*freeing* thread — so gating per-thread confines the extra RSS to the RT
thread's own working set instead of switching reclaim off process-wide:

- **`madvise`**: demoted at the single choke point inside `deallocate_chunk`
  (all four release paths funnel there, so one gate covers every one). Only
  step 3 is skipped — header clear, `back_offset` clear and claim-bit release
  still happen, so the chunk stays immediately recyclable with its pages warm.
- **`munmap`** (§19 large tier): parked on a lock-free pending stack whose
  link lives in the block's own dead meta page — **no allocation on a free
  path**. `kame_pool_rt_pending_bytes()` reports the VA so held.
- **`kame_pool_rt_drain()`** settles both (pending unmaps → this thread's L1 →
  global L2), clearing its own RT flag for the duration so the releases
  actually reach the kernel instead of being re-deferred by the gate that
  queued them.

**Measured while testing, worth recording:** in *steady state* the madvise
gate is never even reached — a freed chunk is parked warm in the §34 recycle
cache and `deallocate_chunk` is not called at all. The gate matters under
cache pressure (over cap / eviction / thread exit). So the original hole was
narrower than stated below, but real. Note also that a thread's L1 keeps the
byte cut it computed when it *armed*, so shrinking the cache cap does not
affect already-armed threads (this is why `alloc_rt_thread_test` drives the
path from fresh threads).

Verified by `tests/alloc_rt_thread_test.cpp` — each claim by observing a
counter move, not by trusting that a call was made: reclaim deferred on an RT
thread but **immediate for identical churn off it** (the per-thread proof),
munmap deferred and reported, `rt_drain` settling to zero.

<details><summary>original G1 write-up (what the hole was)</summary>

§30 gates the *background* and *thread-exit* paths only.  A live-thread
`free()` in steady state can still reach page-reclaim syscalls:

- `deallocate_cold(p)` → `deallocate_chunk(…, reclaim_pages=true default)`
  (`allocator.cpp:3772`, `:3838`) → `madvise(MADV_FREE/DONTNEED)`
  (`:3264/:3273`) when a dedicated/large chunk empties;
- LRC eviction/overflow → `recycle_release_chunk` / `bucket_release_chunk`
  (`:3972`, `:4013`) → same madvise;
- large-tier (§19) free = radix clear + **munmap of the whole mapping**
  (`:7207ff`) whenever the block does not land in (or overflows) the LRC.

With a stable, pre-warmed working set none of these trigger — but
working-set churn makes them reachable, so the current contract silently
depends on an *unstated* "working set never shrinks" assumption.

**Work:** RT-gate these sites.  In RT mode, park-don't-reclaim
(documented RSS growth) **or** defer reclaim onto an explicit
`kame_pool_rt_drain()` the application calls outside its critical loop.
This is the single highest-value change.
</details>

### G2 — prewarm API + violation counters — **DONE**

- **`kame_pool_prewarm(sizes, counts, n)`** allocates, **page-touches** and
  frees each class, holding `counts[i]` blocks concurrently so that much
  capacity is genuinely forced into existence.  Touching is the point: the
  previously-documented allocate/free idiom leaves pages mapped-but-unfaulted,
  so the first RT write still took a minor fault.  Call per RT thread (the
  allocator TLS is per-thread).
- **`kame_pool_rt_violations()`** counts the times an RT thread actually
  entered the kernel for a *new* mapping — the number a test asserts is zero.
  `rt_deferred_reclaims` / `rt_deferred_unmaps` / `rt_pending_bytes` are
  informational (RT mode working as designed, and the growth traded for it,
  visible rather than silent).
- **`kame_pool_set_rt_os_policy()`** — `ALLOW` (default, backward-compatible)
  / `COUNT` / `FAIL` (refuse; the allocation degrades to libc) / `ABORT`
  (report the site and abort — CI/debug).
  **Coverage is deliberately partial:** FAIL/ABORT are honoured at the region
  mmap and the large-tier mmap, both of which have a safe nullptr→libc path.
  A **radix leaf is counted but never refused** — without its leaf a region
  goes unregistered and later frees of pointers inside it would be routed to
  libc `free()`, i.e. corruption rather than degradation.
- **`kame::rt_section`** (C++ RAII) marks the thread, nests correctly
  (restores the previous flag rather than clearing it), and in debug builds
  reports its own violation delta on exit.

### G3 — pre-reserve — **DONE** (`reserve_regions` + `FAIL`)

`kame_pool_reserve_regions(n, prefault)` creates `n` fully-published regions
up front through `mmap_new_region` (metadata init → `radix_insert` → region
list push), optionally touching every slot page.  Combined with
`KAME_RT_OS_FAIL` this *is* the "commit up front, then never map again" mode:
reserve, then any further mapping attempt is refused and degrades to libc
instead of stalling the RT path.  Regions never unmap, so the reservation is
permanent.  Prewarm remains the recommended primary tool (it also reaches the
radix and TLS-bootstrap paths); this is the belt-and-braces option for when
the working-set *size* is known but its size classes are not.

<details><summary>original G2 write-up</summary>

### G2 — pre-warm is an idiom, not an API, and nothing *enforces* it

Cold-claim on the alloc path is inherently unbounded: fresh-region mmap
(`allocator.cpp:6386`), radix-L2 leaf mmap/VirtualAlloc (`:6284/:6279`),
`claim_chunk` swarm-retry (`:3940`), and the per-thread TLS bootstrap
(first malloc on a new thread self-allocates its ~32 KiB TLV block).

**Work:**
1. `kame_pool_prewarm(const size_t sizes[], const unsigned counts[])`
   (per-thread): TLS bootstrap + radix-path touch + chunk pre-claim +
   **page touching** (allocate+free alone leaves pages unfaulted — the
   idiom in `contrib/README.md` under-delivers on first-touch faults).
2. **RT-violation counters**: count inline mmap / madvise / munmap /
   cold-claims while RT mode is on, readable via `kame_pool_get_stats()`.
   Turns the contract from "hope" into a measurable — an RT test asserts
   the counter stayed 0 across the critical section.

</details>

### G4 — the per-op bound, stated with its assumptions — **DONE (as a statement, not a proof)**

**Lock-free ≠ wait-free**, and no amount of prose changes that: a CAS loop has
no *unconditional* iteration bound. What can be stated honestly is a bound that
separates into two kinds, and it is worth keeping them apart because only one of
them is a property of this code alone.

#### (i) Static bounds — hold unconditionally

| Loop | Bound |
|---|---|
| FS=false `allocate_pooled` walk (`:1580`) | **walk-once-bail**: at most `m_count` bitmap words per call, then the tier is abandoned. A compile-time-shaped bound. |
| Word-cache alloc (default ON) | 1 CAS to steal a word, then **one `ctz`** per allocation until the word is spent. Steal frequency is 1/64 allocations. |
| `CrossDeallocBatch` flush | ≤ `CAP` = 1024 entries. Bounded, but 1024× the average — which is why an RT thread bypasses it entirely (G5(b)). |
| Deferred-unmap backlog | ≤ `rt_pending_cap` bytes; settlement is ≤ **one** block per non-RT free (G5(a)). |
| Orphan-chain push/pop | Thread-exit path only; adoption pops **one** node. |

#### (ii) Interference-conditional bounds — need an assumption about the *system*

The bitmap-word CAS loops (alloc `:1373/:1405`, free `batch_clear_impl` `:1962`)
retry only on CAS failure, and **every failure is another thread's success on
that same 64-slot word**. So retries per operation are bounded by the number of
*successful interfering operations on that word* during our attempt — never by
anything unbounded internal to the loop. Converting that into wall-clock WCET
requires a bound on interference, which is a property of the *task set and the
machine*, not of the allocator:

- On a uniprocessor with priority scheduling, interference is bounded by the
  number of preemptions by higher-priority tasks — the classical lock-free RT
  result (Anderson–Ramamurthy–Jeffay, *Real-time computing with lock-free
  shared objects*, 1997), which is why lock-free objects are schedulable there
  at all.
- On a multiprocessor, interference is bounded by the number of contenders
  actually executing, i.e. by the core count and the peers' op rate. This is
  where a hard bound needs a system-level argument (partitioning, or bounding
  the peers' allocation rate) that we do not make.

The same shape covers the seqlock `lookup_chunk` reader retry: its writers are
**cold-path only** (chunk claim / release), so under the RT contract
(prewarmed → no cold claims) the expected retry count is *zero*, not merely
bounded. That is a conditional guarantee, and the condition is exactly the G10
precondition.

#### What is therefore claimed — and what is not

**Claimed:** every loop on the alloc/free path is either statically bounded (i)
or bounded by interfering successes (ii); nothing waits, sleeps, yields, or
takes a lock; and the deferred-work queues are capped. Under the G10
preconditions the syscall count on the free path is **zero**, which is checkable
at runtime (`rt_violations() == 0 && rt_forced_releases() == 0`).

**Not claimed:** a numeric WCET. There is no machine-checked bound, no
wait-free variant, and no static-analysis budget. The G7 tails are *empirical
evidence for a specific machine and load*, not a proof — and their small-op
figures are additionally limited by a ~41.7 ns clock floor (see G7).

**If a hard bound is ever required** (avionics/automotive-grade), the route is
the one this document declined in §1: a **fixed arena** (TLSF-style, P1), where
the allocator never touches the OS and the free lists are private, plus
bounded-retry + a per-thread emergency reserve so a contended CAS can give up
rather than retry. That is a different allocator design, not a tuning of this
one.

### G5 — bound the work one operation inherits — **DONE**

Audited the three amortization sites; each is now bounded or shown
unreachable for a prewarmed realtime thread.

**(a) The realtime deferral backlog itself — was genuinely unbounded.**
Measured: 40 deferred 300 MiB frees parked **12.6 GB** of VA, recoverable only
by an explicit `rt_drain()`. Deferring without a ceiling swaps a bounded
`free()` tail for unbounded memory — the same trap in the other direction.
Fixed by two bounds: a **pending cap**
(`kame_pool_set_rt_pending_cap`, default 1 GiB = what the recycle cache is
already allowed) past which a realtime free releases inline and
`rt_forced_releases()` counts it; and **opportunistic settlement**, where each
non-realtime large free releases at most **one** parked block, so a
mixed-thread program drains itself. Re-measured: 944 MB instead of 12.6 GB,
and zero once ordinary threads run. Pushes stay lock-free; *pops* take an
exclusive gate, because a Treiber pop must read `head->next` and that word
lives in the page a concurrent popper may already have unmapped.

**(b) Cross-thread dealloc batch — bounded, and now bypassed in RT mode.**
`CrossDeallocBatch` accumulates `CAP = 1024` entries (16.4 KiB, L1d-resident
by design) and then **one** free pays for sorting, adjacent-merging and
CAS-ing the whole buffer: a per-op worst case ~1024× the average. Bounded, but
not acceptable inside a deadline. A realtime thread now takes the existing
single-slot `push_direct` path for every ALIGN class, so its free does its own
bitmap CAS and nothing else. This matters because cross-thread free *is* the
STM shape — a Payload cloned on one thread and released on another.
Consequence to know: entries pushed before the section stay unflushed for its
duration (≤ 1024 slots, ≤ ~240 KB — bounded, settled by ordinary activity or
at thread exit), and throughput drops slightly for RT threads in exchange for
the bound.

**(c) `orphan_chain_scrub` — unbounded, but unreachable when prewarmed.**
It walks the whole orphan chain and **restarts from the head whenever its
unlink CAS loses**, so it is O(chain) and worse under contention. It is called
only from `allocate_chunk_path` immediately before mmap'ing a fresh region —
i.e. on the cold claim path that prewarm removes and `KAME_RT_OS_FAIL` refuses
outright. Left as-is deliberately: bounding it would cost throughput on the
path where it pays for itself (it is avoiding an mmap), and no realtime thread
that honours the contract reaches it. **This is a contract dependency, not a
proof** — it belongs in the G10 write-up as an explicit precondition.

**Honest gap in the measurement:** `bench_rt_wcet` has the measured thread
allocate *and* free its own blocks, so it never exercises the cross-thread
class (b) at all. The bound there rests on code reading, not on a measured
tail. A cross-thread arm (thread A allocates, thread B frees) is the obvious
next addition to the harness.

### G6 — page-fault class — **(b) DONE; (a)/(c) remain**

How the field handles this, since it is the one area where every allocator has
had to take a position:

| Concern | glibc | jemalloc | mimalloc | tcmalloc | TLSF / RT practice | kamepoolalloc |
|---|---|---|---|---|---|---|
| **THP** | `glibc.malloc.hugetlb` tunable | `opt.thp = always/auto/never`, `metadata_thp` — explicitly *disableable* | `allow_large_os_pages` (opt-in), `reserve_huge_os_pages_N` | hugepage-aware **by design** (Temeraire, OSDI'21) — manages huge granularity itself rather than leaving it to khugepaged | n/a (fixed arena) | `MADV_HUGEPAGE` opt-in only → **gap (a)** |
| **Don't return pages** | `M_TRIM_THRESHOLD`, `M_MMAP_MAX=0` | `dirty_decay_ms:-1`, `muzzy_decay_ms:-1`, `retain` | `purge_delay=-1` | release-rate knobs | n/a | §30 + §75 per-thread gate |
| **Explicit purge** | `malloc_trim()` | `arena.N.purge` | `mi_collect()` | `ReleaseMemoryToSystem` | n/a | `rt_drain()` |
| **Prefault / pre-commit** | application | **no API** (application) | `reserve_os_memory(commit=true)`, `eager_commit` | prealloc paths | application touches its arena | **`prewarm()` page-touches per size class; `reserve_regions(prefault)`** |
| **Page pinning** | application (`mlockall`) | application | application | application | application (`mlockall` + stack prefault) | **`mlock_regions()` — pool-scoped** |

Two observations from that survey:

* **No allocator prefaults by size class** — it is universally left to the
  application. `mi_option_reserve_os_memory(commit=true)` is the nearest thing,
  and it commits pages rather than provisioning size-class capacity. So
  `kame_pool_prewarm()` is ahead of the field here, not catching up.
* **The RT world's recipe is uniform**: `mlockall(MCL_CURRENT|MCL_FUTURE)` +
  "don't return memory" + "don't grow" (ROS 2's real-time tutorial and the
  `pendulum_control` demo; the PREEMPT_RT / cyclictest community does the same).
  It is established as *application* practice, not an allocator feature.
* THP splits the field: jemalloc offers `never` because a 2 MiB huge page held
  by one small live allocation bloats RSS; tcmalloc went the other way and
  manages hugepages deliberately for TLB. **For realtime, jemalloc's side is
  the right one** — khugepaged/compaction can stall an unrelated fault.

#### (b) Pool-scoped page pinning — **DONE**

`kame_pool_mlock_regions()` / `kame_pool_munlock_regions()` walk the per-NUMA
region lists and `mlock` / `VirtualLock` each 32 MiB region, returning the byte
count actually pinned (a short return = `RLIMIT_MEMLOCK` / working-set quota
reached partway, reported rather than fatal).

This is the one place the pool can beat established practice rather than match
it: `mlockall(MCL_FUTURE)` is the only tool an application otherwise has, and it
pins **every future mapping by every thread**, so one background worker's large
buffer blows the RSS budget. We keep a ledger of our own regions, so we can pin
exactly the pool.

Verified: 5 regions → 167,772,160 bytes pinned, exactly equal to
`reserved_bytes`; `maxrss` moved 3.4 MB → 165.8 MB, i.e. `mlock` genuinely
**populated** the range (so it subsumes `prefault` for the regions it covers),
and `munlock` returned the same count. Covered by
`alloc_rt_thread_test` sub-test (2d), written to tolerate a low quota — CI
containers often cap `RLIMIT_MEMLOCK` at a few MiB — asserting consistency
rather than success.

#### (a) `MADV_NOHUGEPAGE` — still open

We have the *pro*-THP knob (`KAME_POOL_HUGEPAGE=1` → `MADV_HUGEPAGE`) and not
the anti-THP one, which is asymmetric for a library that now claims realtime
support. What is missing: `madvise(MADV_NOHUGEPAGE)` over the slot range of each
region, ideally default-on once a thread is marked realtime (or at least
available as a knob). Linux-only — macOS has no THP. Cost is TLB pressure on
large working sets, which is exactly why it must not be unconditional.

#### (c) The application's half of the checklist — documented, not code

Written into the README contract's exclusions: the realtime thread's **stack**
must be pre-faulted (touch a worst-case local array, or recurse), **code pages**
are demand-paged from the binary on first execution — a *major* fault, possibly
disk I/O — so the loop body should be warmed once before going live, and
memory owned by other libraries (Qt, libc, the driver stack) is outside our
ledger entirely. Stated as scope honesty: `mlock_regions()` covers pool memory,
and pool memory only.

### G7 — WCET tail harness — **DONE** (`tests/bench/bench_rt_wcet.cpp`)

Per-op harness: measured thread (best-effort priority elevation — macOS
`QOS_CLASS_USER_INTERACTIVE`, Linux `SCHED_FIFO` where permitted) + N
interferers, per size band, **RT arm vs OFF arm interleaved per repetition**
(alternating which goes first, so neither owns the warmer cache), log-scale
histogram (4 buckets/octave, allocation-free), reporting **max and
percentiles — never only a mean**, which is the statistic that hides a
syscall.  A percentile is printed only when the sample count can support it
(≥ 10 samples beyond it).  Registered as a ctest smoke; `--full` /
`--pressure` for measurement.  Hard assertion: `rt_violations() == 0`
(machine-independent) rather than any absolute latency.

#### Result 1 — the RT gate, measured (Apple M3, Release, 3 interferers)

`--pressure` measures the > `LRC_HI` (256 MiB) band, where the recycle cache
is bypassed *by construction* so every free really reaches `munmap`.  With
n = 900 samples/arm:

| arm | free p50 | free MAX |
|---|---:|---:|
| **RT** (deferred to `rt_drain`) | **128 ns** | **792 ns** |
| OFF (inline `munmap`) | 20,480 ns | **677,917 ns** |

**160× better median, 856× better tail.** The OFF arm's 678 µs outlier alone
would blow a 1 kHz control budget; `deferred_unmaps = 900` confirms all 900
frees took the deferred path. This is the empirical validation of G1.

Honest counter-entry: the RT arm's *malloc* is ~27 % slower on the mean
(6.4 µs vs 5.1 µs) because holding the VA until the drain means fresh
mappings instead of reuse. The trade is explicit — a slightly worse mmap
mean for a 160×–856× better free tail — and a real RT design would not be
mapping 300 MiB inside the loop anyway.

#### Result 2 — the structural finding (more important than Result 1)

For **every band at or below `LRC_HI`** the two arms are *statistically
identical* (p99.9 equal to the nanosecond), because the recycle cache absorbs
the release outright: no `madvise`, no `munmap`, nothing to defer.  Verified
by probe that this holds **even with the cache cap forced to zero** — zeroing
the cap is *not* a way to manufacture pressure (a chunk still lands in the
per-thread L1, whose byte cut is fixed when the thread *arms*, and the
smallest size class fits at idx 0 regardless).

So in steady state the allocator is already syscall-free across 1 B – 256 MiB,
and the RT gate is a **safety net** for three specific regimes rather than a
steady-state necessity:
  1. the > `LRC_HI` band (Result 1);
  2. cross-thread / fresh-thread release patterns (covered by
     `alloc_rt_thread_test`, where a fresh thread under a zeroed cap *does*
     reach the `madvise` path);
  3. thread exit (§21, already gated by §30).

Caveat on the deep tail: p99.9999 needs ~10⁷ samples, which the huge band
cannot reach in reasonable time — the harness prints only the percentiles its
sample count supports, so no reported figure is an artefact of a single
outlier. The small bands do reach p99.99+ under `--full`.

Still owed here: G9's manifesting regression test, on Linux (see G9).

### G8 — §74 single mmap+radix site — **DONE, no work remaining**

Already landed in `c04a7975d`: `allocate_chunk<ALLOC>()` no longer carries
its own region walk / mmap / bitmap-CAS — it is LRC-pop → `claim_chunk`
→ header-stamp, and `mmap_new_region()` has exactly one caller
(`claim_chunk`, `allocator.cpp:3957`).  The `(§74) The SINGLE region-walk
+ mmap + bitmap-claim site` note at `allocator_prv.h:1031` is a statement
of the achieved state, not a TODO.  See §1.1 for the resulting four-site
audit surface — G1–G3 gate those directly, with no refactor first.

### G9 — teardown L1-stranding — **fix and test both DONE; only a Linux negative control is owed**

Fix landed in `3145e139a`: both L1 entry points gate on
`kame_thread_torn_down()` — `l1_push` (beside the pre-existing `s_l1_drained`
check, so never-armed threads are covered too) and `l1_pop_fit` (no re-arm,
closing the symmetric `g_lrc_l1_threads` counter drift). Verified present in the
current tree (`allocator.cpp:6803`, `:6770`).

The "manifesting regression test" that commit said it owed **also already
exists**: `tests/alloc_thread_exit_unarmed_test.cpp` forces exactly the
sufficient condition — a producer allocates a 48 KiB dedicated block and hands
it to a consumer thread that performs **no allocations of its own** (so its L1
is never armed and `s_l1_drained` stays false), whose only pool interaction is
freeing that foreign block **from its own `pthread_key` destructor** at thread
exit. The cycle repeats and asserts `chunks_live` / `units_live` plateau instead
of growing +1/cycle. Built both ways (static + `_dynamic`) and registered in
ctest, both passing.

**What is actually still owed is a *negative control*, on Linux.** The test
passes on macOS, but macOS *cannot* trigger the bug (glibc runs C++
`thread_local` destructors before `pthread_key` destructors; dyld's order does
not open the window), so a pass there proves nothing about the test's teeth. The
outstanding question is narrow and answerable: **with the `3145e139a` guard
reverted, does this test fail on Linux?** If it does not, the test does not
cover the fix and needs strengthening. See `design/RT_LINUX_HANDOFF.md`.

### G10 — the RT contract, stated in the README

One section: *given* realtime mode + prewarm + stable working set +
bounded threads, malloc/free in [1 B .. 32 MiB] performs no syscalls and
completes in a bounded number of atomic steps; sizes > 32 MiB and
working-set growth are excluded; violations are observable via the G2
counters.

## 3. Suggested order

**Done:** G8 (`c04a7975d`) and G9 (`3145e139a`) were already landed before
this audit; G1 + G2 + G3 landed with the §75 realtime work
(`tests/alloc_rt_thread_test.cpp`).

G7 (the tail harness) followed, and its numbers are in that section.

**Remaining:**

```
G6(a)  MADV_NOHUGEPAGE for the arena (Linux only; we ship the
       pro-THP knob and not the anti-THP one, which is
       asymmetric now that realtime is claimed)
G9-nc  Negative control on Linux: revert 3145e139a and confirm
       alloc_thread_exit_unarmed_test actually FAILS
```

Everything else is done and measured on both the same-thread and cross-thread
paths, with the claims and their exclusions written down (G4, G10).  Both
remaining items need a Linux host — handoff in `design/RT_LINUX_HANDOFF.md`.
