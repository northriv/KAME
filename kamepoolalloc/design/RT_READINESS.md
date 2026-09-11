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

**(b) Cross-thread dealloc batch — bounded by a per-thread flush threshold.**
`CrossDeallocBatch` accumulates `CAP = 1024` entries (16.4 KiB, L1d-resident by
design) and then **one** free pays for sorting, adjacent-merging and CAS-ing the
whole buffer: a per-op worst case ~1024× the average. Bounded, but not
acceptable inside a deadline. Marking a thread realtime sets **that thread's
batch `cap` to 1**, so each free settles its own slot. This matters because
cross-thread free *is* the STM shape — a Payload cloned on one thread and
released on another.

Measured, 4 M samples per arm (32 B, producer allocates / measured thread
frees):

| | p99.9 | p99.99 | p99.999 | MAX |
|---|---:|---:|---:|---:|
| **RT** (`cap = 1`) | **96 ns** | **256 ns** | 10,240 ns | 31,791 ns |
| OFF (batched, `CAP` 1024) | 1,792 ns | 10,240 ns | 16,384 ns | 96,709 ns |
| *(first attempt, `push_direct`)* | *1,792 ns* | *7,168 ns* | *14,336 ns* | *27,000 ns* |

**19× at p99.9, 40× at p99.99**; max 3× (97 µs → 32 µs, but that is one sample
— read the percentiles).

Two things were learned getting here, both by measurement, and the first
attempt's row is kept because the contrast is the point:

* **The first attempt cost 4 % of throughput.** Routing realtime threads to
  `push_direct` via an `if(g_rt_thread)` test in `deallocate_pooled` put a
  `thread_local` read on the cross-thread free hot path — and on a macOS dylib
  that is a `_tlv_get_addr` **call**, on a ~3.5 ns operation. Interleaved A/B
  against the pre-§75 tree: **285.9 → 271.8 M free/s, base winning 6/6 reps**;
  deleting just that branch recovered it (5/5). Moving the decision into the
  batch object — which `push` already dereferences — makes it free. Residual is
  −1.75 % at 1/10 reps, and a control that removed even the `cap` load measured
  *slower* (−6.4 %), which is physically implausible and marks where this
  benchmark's attribution power ends.
* **`push_direct` was never reliably direct.** It is *adaptive*: it reads the
  chunk's coalesce hint and routes to HOLD above a threshold, with an
  epsilon-greedy explore that force-holds periodically. So the first fix still
  accumulated and still spiked — which is exactly why its p99.9 came out
  *identical* to the batched arm. Only the unconditional `cap = 1` removes the
  mid-tail.

Consequence to know: entering STRICT flushes whatever ordinary work left in the
batch (otherwise the section's first free would inherit up to `CAP` entries), so
mark the thread once rather than per cycle if that boundary cost matters.

**This is why the level is split.** Measuring the throughput cost of the batch
change on the STM shape gave **−48 %** (6/6 reps, 60.8 → 31.2 M free/s), against
**−2.8 %** for the syscall deferral alone (median; individual reps came out
*positive*, i.e. within noise). Bundling them would have forced every soft-realtime
caller to buy a p99.9 improvement they cannot use at half their free throughput.
So `KAME_RT_DEFER` is the cheap half and can be a process-wide default
(`kame_pool_set_realtime_default`), while `KAME_RT_STRICT` stays per-thread —
its cost is on a hot path, so it is never implied and never global. KAME itself
enables DEFER process-wide and deliberately does not enable STRICT: its deadlines
are instrument I/O at millisecond scale.

**(c) `orphan_chain_scrub` — unbounded, but unreachable when prewarmed.**
It walks the whole orphan chain and **restarts from the head whenever its
unlink CAS loses**, so it is O(chain) and worse under contention. It is called
only from `allocate_chunk_path` immediately before mmap'ing a fresh region —
i.e. on the cold claim path that prewarm removes and `KAME_RT_OS_FAIL` refuses
outright. Left as-is deliberately: bounding it would cost throughput on the
path where it pays for itself (it is avoiding an mmap), and no realtime thread
that honours the contract reaches it. **This is a contract dependency, not a
proof** — it belongs in the G10 write-up as an explicit precondition.

**Methodology notes**, both of which cost a wrong conclusion once:
  * At 120 k samples the RT arm looked *worse* at p99.9; at 4 M it wins by 19×.
    This comparison needs ~10⁶ samples — below that the deep-tail buckets hold
    single digits and the ordering is noise.
  * `p50 = 0 ns` is not a sub-nanosecond free, it is the **clock floor**:
    `steady_clock` on Apple Silicon ticks at ~41.7 ns, so per-op latencies below
    that quantize to {0, 42}. Only tails ≳ 100 ns are meaningful here.

### G6 — page-fault class — **DONE** ((a), (b) code; (c) documented)

How the field handles this, since it is the one area where every allocator has
had to take a position:

| Concern | glibc | jemalloc | mimalloc | tcmalloc | TLSF / RT practice | kamepoolalloc |
|---|---|---|---|---|---|---|
| **THP** | `glibc.malloc.hugetlb` tunable | `opt.thp = always/auto/never`, `metadata_thp` — explicitly *disableable* | `allow_large_os_pages` (opt-in), `reserve_huge_os_pages_N` | hugepage-aware **by design** (Temeraire, OSDI'21) — manages huge granularity itself rather than leaving it to khugepaged | n/a (fixed arena) | **`kame_pool_set_thp_policy()` — `SYSTEM`/`ALWAYS`/`NEVER`, and re-advises regions already mapped** |
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
* **A `PREEMPT_RT` kernel has no THP at all, so (a) is a general-purpose-kernel
  knob.** Upstream `mm/Kconfig` gates `TRANSPARENT_HUGEPAGE` on
  `HAVE_ARCH_TRANSPARENT_HUGEPAGE && !PREEMPT_RT`, and Ubuntu's
  `7.0.0-29-realtime` confirms it: the symbol is *absent* from the config
  (`/sys/kernel/mm/transparent_hugepage/enabled` does not exist) while the
  matching generic build has `CONFIG_TRANSPARENT_HUGEPAGE=y`.  Coherent —
  khugepaged collapses are the class of spike such a kernel exists to remove.
  Consequences: the fault-path spike (a) suppresses **cannot occur** on an RT
  kernel, so the knob is aimed at soft-realtime work on a stock kernel;
  `kame_pool_set_thp_policy()` there is a silent no-op (`MADV_NOHUGEPAGE`
  returns `EINVAL`, and the re-advise walk's `0 MiB` cannot be told apart from
  "nothing to re-advise"); and (a)'s evidence must stay the generic-kernel
  measurement below, since the arms are unmeasurable on an RT host.  Note also
  that Ubuntu's generic build is `TRANSPARENT_HUGEPAGE_MADVISE`, not
  `_ALWAYS` — unadvised ranges get no hugepages there either, so `SYSTEM` ≈
  `NEVER` and the only informative A/B is `ALWAYS` against `NEVER`.
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

#### (a) `MADV_NOHUGEPAGE` — **DONE**, opt-in, measured

`kame_pool_set_thp_policy(int)` / `kame_pool_get_thp_policy(void)`, with
`KAME_THP_SYSTEM` (0, default) / `KAME_THP_ALWAYS` (1) / `KAME_THP_NEVER` (2).
This replaces the old env-only `KAME_POOL_HUGEPAGE` read; that variable, plus
a new `KAME_POOL_NOHUGEPAGE`, now merely seeds the initial value for
`LD_PRELOAD` use, and an explicit call always wins.

Three things were needed beyond the two-line mirror of the `MADV_HUGEPAGE`
block:

* **Regions are created lazily**, so a policy set after the first allocation
  would miss everything already mapped. `PoolAllocatorBase::thp_advise_regions()`
  walks the per-NUMA region lists (the same walk as `mlock_regions()`) and
  re-advises; `set_thp_policy` returns the bytes it covered, so the caller can
  see the walk reached something. Measured: with 6 regions reserved but
  untouched and the policy set afterwards, `AnonHugePages` over those regions
  is **202,752 kB (100 % of Rss) by default vs 0 kB under `NEVER`** — the walk
  is doing the work, not the per-claim advise.
* **The large-VA tier needed it more than the regions did.** `mmap_new_region`
  was the only site the handoff note identified, but blocks above `LRC_HI` are
  a *fresh 32 MiB-aligned mmap on every allocation* and are the coldest, largest
  memory the pool hands out. Advising only regions left every one of those spans
  faulting as a hugepage; `large_va_raw_map` now carries the same policy. This
  was caught by measurement, not review — see the fault numbers below.
* **`KAME_THP_SYSTEM` cannot be re-applied to an already-advised region.**
  Linux has no "clear" advice: `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE` each clear
  the other's VMA flag, neither restores the neutral state. Policy 0 therefore
  returns 0 from the walk and applies to new regions only. Documented rather
  than papered over.

Also worth stating, because it changes the recommended call order: **`NEVER`
does not split hugepages that already exist.** It prevents future hugepage
faults and future khugepaged collapses. So set the policy *before* prewarm,
not after — `set_thp_policy(NEVER)` → `prewarm` → `mlock_regions`. Pages
already huge are already resident, so they are an RSS and TLB fact rather than
a latency one, but the RSS is not small: THP inflated the same working set
from 152,868 kB to 202,752 kB (+33 %) in the measurement above.

##### Measured: what it buys and what it costs

Host: 4 vCPU Intel Xeon @ 2.80 GHz (Firecracker microVM, avx512, 16 GiB),
Ubuntu 24.04, glibc 2.39, kernel 6.18.5, GCC 13.3, Release **x86-64** (the
`-m32` build is covered separately below),
`transparent_hugepage/enabled = always`, `defrag = madvise`. A noisy shared
VM: every figure is a **median of 9 interleaved cross-process repetitions**
(the arms cannot be interleaved *within* a process, since `NEVER` does not
split existing hugepages).

**This host is NOT a realtime kernel, and it matters for how far the numbers
below can be pushed.** `PREEMPT_DYNAMIC`, not `PREEMPT_RT`
(`/sys/kernel/realtime` is absent); no `isolcpus`, `nohz_full` or `rcu_nocbs`;
`sched_rt_runtime_us` left at the default 950 ms/1 s, so even the
`SCHED_FIFO prio 80` the harness reports getting is throttled at 95 % and
preemptible; and it is a 4-vCPU guest, so steal time is in every sample.
Split the results accordingly:

* **Mechanism — trustworthy here.** Whether `MADV_NOHUGEPAGE` stops the kernel
  zeroing 2 MiB at a time is a property of the page-fault path, not of
  scheduling. The p50 moving 27 ns → 2,048 ns (every page now paying its own
  4 KiB fault instead of 511 of 512 riding free on a hugepage) and
  `AnonHugePages` going 100 % → 0 are direct observations of that mechanism,
  and the `> 8 µs` sample count matching the 2 MiB span count exactly is the
  arithmetic confirming it.
* **Absolute WCET — NOT trustworthy here.** The `MAX` column, and to a lesser
  extent `p99.99`, measure this hypervisor and this scheduler at least as much
  as they measure the allocator. The same-arm max ranging 0.98–32.8 ms across
  repetitions is the tell. Treat those cells as upper bounds contaminated by
  the environment, and re-take them on a `PREEMPT_RT` host with the realtime
  thread on an isolated, `nohz_full` core before quoting any of them as a
  worst-case execution time.

The *ratios and directions* are what this environment can establish, and they
are what the opt-in decision below rests on; nothing here is offered as a
measured WCET bound.

**The tail** — `bench_rt_wcet --faults`, a new mode added for this: one timed
write per 4 KiB page across freshly-mapped `> LRC_HI` memory, 196,608 samples
per run.

| policy | mean | p50 | p99 | p99.9 | p99.99 | MAX |
|---|---:|---:|---:|---:|---:|---:|
| `SYSTEM` (default) | 2,580 ns | 27 ns | 320 ns | 459 µs | 918 µs | 32.8 ms |
| `ALWAYS` | 1,268 ns | 27 ns | 384 ns | 524 µs | 918 µs | 19.9 ms |
| **`NEVER`** | 2,352 ns | **2,048 ns** | **6,144 ns** | **41 µs** | **98 µs** | **224 µs** |

Read it as a redistribution, not a win: under THP 511 of every 512 page
touches are free and the 512th zeroes 2 MiB; under `NEVER` every page pays its
own ~2 µs fault. The **mean is a wash** (2,580 vs 2,352 ns), the p50 gets 95×
*worse*, and the deep tail gets **11× better at p99.9, 9× at p99.99, and two
orders of magnitude at the max**. That is exactly the trade a realtime caller
wants and a throughput caller does not. The count of samples above 8 µs makes
the mechanism visible: 24 rounds × 16 hugepage spans ≈ 400 such samples under
THP, matching the 2 MiB span count almost exactly.

**The throughput cost** — the allocator does not care; the application does.

| bench | default | `NEVER` | |
|---|---:|---:|---|
| `bench_loop_pool` 64 B / 4 KiB / 64 KiB (M ops/s) | 272 / 151 / 74 | 271 / 162 / 75 | **neutral** (0.99×–1.07×) |
| `bench_tlb` 32 MiB working set (ns/hop) | 111.9 | 128.0 | +14 % |
| `bench_tlb` 128 MiB | 115.3 | 151.8 | +32 % |
| `bench_tlb` 512 MiB | 128.9 | 203.3 | **+58 %** |

`bench_tlb` (new, `tests/bench/bench_tlb.c`) is a dependent random pointer
chase over a pool-allocated working set — deliberately the *worst* case for
TLB reach, with no memory-level parallelism to hide a page walk behind. Real
workloads sit between it and zero. The existing benches could not see this at
all: `bench_loop` keeps one block live, and `bench_rt_wcet` measures the
allocator rather than the application's access to what it allocated.

##### Should `set_realtime_mode(1)` imply policy 2? — **No**

Decided after measuring, and the measurement supports the prior. A knob
documented as "silences background maintenance" that also slowed a 512 MiB
working set by 58 % would be a genuinely surprising side effect, and the win it
buys (a bounded first-touch fault) is only reachable by callers who also
prewarm and pin — i.e. callers who are already reading this section and can
make the call themselves. It stays opt-in, one explicit line in the realtime
recipe.

##### Test

`alloc_rt_thread_test` sub-test (2e): API round-trip, out-of-range rejection,
the re-advise walk covering whole regions, and — on a fresh `> LRC_HI` block —
`AnonHugePages` held at 0 under `NEVER`. It establishes a **baseline first**
and skips the behavioural half if the host backs no transparent hugepages at
all, so it cannot pass for the wrong reason. Negative control run the same way
as G9's: with the `large_va_raw_map` advise disabled the sub-test FAILS
(10,240 kB instead of 0), so it has teeth.

The baseline arm is `KAME_THP_ALWAYS`, deliberately, not `KAME_THP_SYSTEM`.
An *unadvised* range is only opportunistically backed by hugepages: under the
common `defrag = madvise` setting the kernel will not compact to find a 2 MiB
block for it, so a `SYSTEM` baseline reads 0 as soon as memory is mildly
fragmented and the check skips itself for no good reason. That is not
hypothetical — with a `SYSTEM` baseline this sub-test passed 64-bit and
skipped 32-bit on the *same host and kernel*. `MADV_HUGEPAGE` does earn the
compaction effort, so `ALWAYS`-vs-`NEVER` is both deterministic and the
sharper test: it exercises each direction of the policy rather than one
direction against the kernel's whim.

Note for anyone re-running this in a container: many sandboxes set
`PR_SET_THP_DISABLE` on the whole process tree, which makes every VMA report
`THPeligible: 0` no matter what `/sys/kernel/mm/transparent_hugepage/enabled`
says. Check `THP_enabled:` in `/proc/self/status`; if it is 0, clear it with
`prctl(PR_SET_THP_DISABLE, 0)` in a wrapper that then `exec`s the benchmark
(the flag is in `MMF_INIT_MASK`, so the cleared state survives `exec`).
Otherwise the (2e) skip and every `NEVER` measurement are vacuous.

Incidentally, `PR_SET_THP_DISABLE` is the third way to get anti-THP, and the
reason it is not what we do: it is process-wide, so it would disable
hugepages for the application's own non-pool memory too — the same
blunt-instrument problem `mlockall(MCL_FUTURE)` has versus `mlock_regions()`
in (b).

##### 32-bit (ILP32)

Checked, because the standalone library claims Linux 32-bit support and this
work touches page-level code. Built `-m32` (GCC 13.3 multilib, same host):
**ctest 18/18, warning-free**, and sub-test (2e)'s behavioural half passes
there too — `MADV_HUGEPAGE` 10,240 kB vs `MADV_NOHUGEPAGE` 0 kB, identical to
the 64-bit figures. Transparent hugepages are a property of the kernel's page
tables, not of the process's pointer width, so an ILP32 process on an x86-64
kernel gets them normally; a plain `mmap` + `MADV_HUGEPAGE` control confirms
it.

Nothing in the policy itself is width-sensitive: the region walk is over the
same `s_region_dll_heads` list, and the `madvise` length is
`(size_t)ALLOC_MIN_MMAP_SIZE - ALLOC_PAGE_SIZE`, cast before the subtraction.
The pool's own ILP32 handling was already in place (`ALLOC_MAX_REGIONS = 96`
= 3 GiB, `RADIX_VA_LIMIT = ~0`).

Two ILP32-only defects were found and fixed by building it, both in the new
test/bench code rather than the library: (2e)'s smaps parser scanned `%lx`
into a `uintptr_t`, which is `unsigned int` on ILP32 (a `-Wformat` warning,
benign where `long` is also 32-bit but wrong in principle — now scans
`unsigned long long` and casts); and `bench_tlb` silently overflowed
`size_t` for a multi-GiB working set, surfacing as a misleading "working set
too small" — it now range-checks and says which it is.

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

G9's negative control, which this section previously flagged as owed, has
now been run on Linux — see G9.

#### Result 3 — what a STOCK allocator's worst case actually is

The survey table above says what each allocator *offers*; this is what they
*do*.  `bench_rt_wcet_malloc` is the same harness built against plain
malloc/free and not linked to kamepoolalloc, so the allocator under test is
whichever one `LD_PRELOAD` supplies.  One harness, one clock, one histogram —
without that a cross-allocator tail comparison is not worth printing.

Host: the non-realtime `PREEMPT_DYNAMIC` Firecracker VM described in G6(a).
`--pressure --reps 5 --iters 2000 --threads 3`, medians of **9 interleaved
repetitions** with the arm order rotated each time.  Every allocator at its
DEFAULT settings — none of the RT knobs from the survey table
(`dirty_decay_ms:-1`, `purge_delay=-1`, …) was applied to any of them,
including ours.

| band | allocator | malloc p99.9 | malloc MAX | free p99.9 | free MAX |
|---|---|---:|---:|---:|---:|
| 64 B | glibc | 192 ns | 142 µs | 96 ns | 160 µs |
| | mimalloc | 96 ns | 24 µs | 80 ns | 32 µs |
| | jemalloc | 12 µs | 149 µs | 12 µs | 253 µs |
| | kamepool | 80 ns | 122 µs | 80 ns | 82 µs |
| 4 KiB | glibc | 192 ns | 74 µs | 112 ns | 85 µs |
| | mimalloc | 384 ns | 28 µs | 192 ns | 31 µs |
| | jemalloc | 16 µs | 220 µs | 14 µs | 210 µs |
| | kamepool | 112 ns | 82 µs | 80 ns | 78 µs |
| 256 KiB | glibc | 192 ns | 42 µs | 112 ns | 25 µs |
| | mimalloc | 640 ns | 40 µs | 320 ns | 20 µs |
| | jemalloc | 16 µs | 72 µs | 16 µs | 83 µs |
| | kamepool | 192 ns | 29 µs | 224 ns | 23 µs |
| **8 MiB** | glibc | 29 µs | 252 µs | 98 µs | 312 µs |
| | mimalloc | **3 µs** | **21 µs** | **320 ns** | **17 µs** |
| | jemalloc | 98 µs | 445 µs | 328 µs | **39.9 ms** |
| | kamepool | 4 µs | 48 µs | 4 µs | 22 µs |
| **300 MiB** | glibc | 33 µs | 385 µs | 164 µs | 688 µs |
| | mimalloc | 41 µs | **64 µs** | **2 µs** | **16 µs** |
| | jemalloc | 115 µs | 1.2 ms | 328 µs | 1.8 ms |
| | kamepool | 57 µs | 169 µs | 98 µs | 236 µs |
| 32 B x-thread free | glibc | — | — | 4 µs | 103 µs |
| | mimalloc | — | — | **768 ns** | **23 µs** |
| | jemalloc | — | — | 1 µs | 84 µs |
| | kamepool | — | — | 10 µs | 66 µs |

Reading it honestly:

* **The small-band MAX column is the environment, not the allocator.**  A
  64 B `malloc` cannot spend 100 µs of CPU; 24–160 µs maxima appear for all
  four, which is the signature of this VM's scheduler and steal time.  Use
  p99.9 in those rows and treat MAX as the noise floor.  The large bands are
  where MAX starts meaning something, because the work there is real.
* **A stock allocator's honest worst case, in one sentence:** sub-microsecond
  at p99.9 for small blocks, and **hundreds of microseconds to milliseconds
  once the block is big enough to reach `mmap`/`munmap`** — glibc 312 µs /
  688 µs on the 8 MiB and 300 MiB frees, jemalloc worse.
* **jemalloc at defaults has the worst tail here by a wide margin** — 39.9 ms
  on an 8 MiB free, and a p99.9 of 12–16 µs even on 64 B.  That is
  decay-based purging (`dirty_decay_ms` / `muzzy_decay_ms`) running on the
  calling thread.  It is also the allocator with the most explicit knob for
  it, so this is a statement about defaults, not about jemalloc's ceiling.
* **mimalloc has the best tail of the stock three, and beats us** on the
  large bands and on cross-thread free.  Worth saying plainly rather than
  burying: on this host it is the allocator to beat, and its 17 µs / 16 µs
  large-block free maxima are better than our 22 µs / 236 µs.
* **kamepool as a plain `LD_PRELOAD` drop-in** — i.e. with none of §75's
  realtime API called — sits with mimalloc on the small bands and between
  mimalloc and glibc on the large ones.

##### And the realtime mode, at these parameters, did not help

Measured the same way with the direct API (`bench_rt_wcet`, RT arm vs OFF arm,
median of 9), the 300 MiB free is 131 µs p99.9 in BOTH arms.  The counter says
why: `deferred_unmaps=15` out of 10,000 frees.  At 300 MiB a block, the G5
pending cap (1 GiB default) is exhausted after three deferrals and every
subsequent free is forced inline — which is G5 working exactly as designed
("bounded memory beats a bounded tail"), and it means the macOS headline
figure (128 ns vs 20,480 ns median free) is only reachable when the cap is
large relative to the block size.  Anyone quoting that number should also
quote `kame_pool_get_rt_pending_cap()`, or raise it for the block size in
play.  In the same runs the RT arm's *malloc* was slower on the 300 MiB band
(p99.9 115 µs vs 57 µs), consistent with the trade already recorded in
Result 1: holding VA means fresh mappings instead of reuse.

#### Result — a real `PREEMPT_RT` host (Ubuntu 26.04, i5-7500, isolated cores)

Everything above was measured on a non-realtime kernel, which is why G7's
absolute numbers carried a caveat.  This is the same harness on a host that
removes it: `7.0.0-29-realtime` (`CONFIG_PREEMPT_RT=y`), cores 2-3 isolated
with `isolcpus`/`nohz_full`/`rcu_nocbs`, all IRQs steered to 0-1, PM-QoS held
at 0 µs, `performance` governor, `sched_rt_runtime_us = -1`, no thermal
throttling.  `--full`, five repetitions, medians.  Setup and provenance are in
`design/RT_LINUX_HANDOFF.md`.

**Cross-thread free** — a producer thread allocates, the measured thread frees:

| | mean | p50 | p99.99 | p99.999 | **MAX** |
|---|---|---|---|---|---|
| **RT** | 55 ns | 55 ns | **160 ns** | **192 ns** | **352 ns** |
| OFF | 43 ns | 31 ns | 20,480 ns | 32,768 ns | **42,356 ns** |
| ratio | 1.3× | 1.8× | **128×** | **171×** | **120×** |

**1.8× on the median buys 120× on the worst case.**  The mechanism is the one
the harness footnotes: OFF batches to `CAP=1024` and one unlucky free then
pays for the whole buffer, while RT takes `push_direct` every time.  Unlike
the 300 MiB result above, the pending cap is not in play here — the band is
32 B — so this is the gate working in the regime it was designed for.

Steady-state maxima, same runs:

| band | RT malloc MAX | RT free MAX |
|---|---|---|
| 64 B (bucket) | 449 ns | 332 ns |
| 4 KiB (bucket) | 415 ns | 301 ns |
| 256 KiB (dedicated) | 673 ns | 237 ns |
| 8 MiB (large) | 16,087 ns | 239 ns |

**Stated with the host's resolution, which is the point of measuring on an RT
box at all**: `hwlatdetect` recorded no sample above 10 µs in 300 s and
`cyclictest` maxed at 12–16 µs, so the bucket-band figures sit a factor of ~30
below the floor and are allocator numbers, while the 8 MiB malloc max sits *at*
the floor and must not be attributed to the allocator.  Quote them together.

**The hard assertion fails on this host, with a characterised cause.**
`rt_violations` is deterministic and `reps`-shaped — 0 / 2 / 2 / 4 / 8 / 18 at
`--reps` 2 / 3 / 4 / 6 / 10 / 20, i.e. `reps − 2` at even counts, with odd ones
not fitting (3 gives 2) — and is unchanged by removing the interferers or by
pinning to four cores.  The registered `bench_rt_wcet_smoke` ctest runs
`--reps 2`, which is exactly the zero of that sequence, which is why CI has
never seen it.  `--rt-os-policy 3` names the site as `large_va_raw_map`, not the radix
leaf — one of the two sites that degrade safely to libc under
`KAME_RT_OS_FAIL`, so the bound is not what breaks.  The 8 MiB band is far
below `LRC_HI`, so the large recycle cache is meant to absorb it: this is a
prewarm/recycle shortfall of one mapping per repetition, open and tracked in
the handoff document rather than papered over here.

**One harness bug was found by running on two cores instead of four.**
`interferer()` and `xt_producer()` are documented as deliberately not
realtime, but `pthread_create` defaults to `PTHREAD_INHERIT_SCHED` and both
are started after the measuring thread promotes itself, so they were running
at `SCHED_FIFO` 80 as well.  Beyond measuring the wrong contention, that
deadlocks outright once runnable threads exceed CPUs — equal-priority FIFO
threads never preempt each other, and `sched_rt_runtime_us = -1` has already
removed the throttle.  It hid for as long as threads ≤ CPUs, which was true of
every host used before.  Fixed by `demote_this_thread()`; **numbers taken
before that fix are not comparable** — the same cross-thread RT max read
1,988 ns under the old regime against 352 ns after.

#### Result — the STM's commit latency under the deployment's roles

G7's bands measure the allocator.  What an acquisition deadline actually asks
about is the *transaction*, and `transaction_priority_mixed_test` now reports
it: the HIGHEST record commit, timed, under the role mix that deployment
actually has (NORMAL driver peers, a UI thread taking root Snapshots, a
SCRIPTING thread), with the acquisition thread at `SCHED_FIFO` 20 on an
isolated core.  120 s, **6,568,736 commits**, on the PREEMPT_RT host:

| mean | p50 | p99 | p99.9 | p99.99 | p99.999 | **MAX** |
|---|---|---|---|---|---|---|
| 800 ns | 768 ns | 2.05 µs | 20.5 µs | 32.8 µs | 81.9 µs | **95.1 µs** |

> **This tail was largely G7's own subject, measured with G7's contract
> unhonoured.**  `transaction_priority_mixed_test` called `prewarm` and nothing
> else — no `set_realtime_mode`, no `set_realtime_thread` — so the acquisition
> thread's **cross-thread** frees ran ungated, batched to `CAP=1024` with one
> unlucky free paying the whole sort+merge+CAS.  A commit frees cross-thread
> exactly when a peer allocated on its subtree, which is why the effect tracked
> the cross-subtree role and nothing else, and why the slow-commit RATE scales
> with payload clones per commit (31.6 → 117.9 per million from 4 to 16 leaves,
> against the 3.4× the clone count predicts) while the MAGNITUDE of an event
> stays fixed.  Adding `kame_pool_set_realtime_thread(KAME_RT_STRICT)`: slow
> commits 28.8 → **8.3** per million, MAX 92.6 → **66.7 µs**, throughput
> **+8 %**.  Mode alone (31.0) and `KAME_RT_DEFER` (29.3) are within noise —
> only STRICT, the one level that drops the batch.  The test now defaults to
> the contract; **KAME still does not mark the thread**, so precondition 3
> remains outstanding in the application.
>
> This block used to end "66.7 µs is *below* the 67.9 µs `latency_floor`
> measures as this host's worst case with no STM at all" — **retracted**.  The
> arms ran *with* isolation and 67.9 µs is `latency_floor`'s *un-isolated* row;
> the right floor for the regime is 17 µs, the same one the next paragraph
> already quoted, two lines apart and contradicting it.  66.7 µs is therefore
> 3.9x the machine, not level with it.

The host's floor is **17 µs** (`rtla osnoise`, 120 s, Max Single, which bounds
C-states, SMIs and `nohz_full` wake-ups in one number) — and 219 ns once
`/dev/cpu_dma_latency` is held at 0, which `rtla` does not tell you.  So the
66.7 µs worst case is **3.9x** the 17 µs floor, and removing the floor
altogether moves it only to 53.1 µs: the tail is the STM's own retry path, not
the machine.  Everything through p99 stays a factor of eight below the floor.

Three things about that number are worth stating with it.

**It requires precondition 2.**  Before the test called `kame_pool_prewarm()`
the MAX was ~400 µs, and immovably so — unchanged across four workload
configurations, two run lengths, isolated versus housekeeping cores, and
PM-QoS held at 0 or not.  It was the pool's §29 freelist pre-fill faulting five
`FS=true` size classes' first chunks in the first commit: 321 page faults at
t=0 and none for the next 56 s.  Not a bound, a cold-start artefact.  **Nothing
in `kame/` or `modules/` calls `kame_pool_prewarm()`**, so that spike is live
in the application; one call on `XPrimaryDriver`'s acquisition thread removes
it, and unlike `KAME_POOL_DISABLE_PREFILL=1` it costs no throughput.

**It is not the negotiation sleep, and not a retry storm.**  The sleep chunk is
1 ms and nothing approached it.  Slow commits (>= 50 µs) averaged 2.08 attempts
with a maximum of 4, and two passes of an 800 ns commit is 1.6 µs — so the
passes are long, ~25 µs each, rather than numerous.  The other roles completed
a mean of 13 commits during each slow one, i.e. their normal rate: nobody was
stuck.  Kernel tracing of such a window finds no syscall, no fault, no context
switch and no IRQ.  Locating the remaining ~25 µs needs instrumentation inside
`commit()`; the kernel cannot see it.

**The priority machinery is what keeps it there.**  `transaction_latency_bench`
on the same host, four symmetric threads with no priority differentiation, is
the control: its leaf band is flat at 384 ns to p99.9 and then jumps to 3.1 ms
at p99.99 with a 32.6 ms max, and 1,217 of its 1,234 slow commits had reached
at least one 1 ms negotiation sleep chunk while the rest of the system
completed a mean of 15,708 commits.  That is a thread losing and sleeping.  The
acquisition thread, at HIGHEST under the deployment's roles, never gets there.

#### …but KAME does not ship HIGHEST, and the shipped arm answers differently

Everything above is the **library's ceiling**.
`XPrimaryDriverWithThread::AcquisitionPriority` is `ScopedPriority(NORMAL)`
plus an OS elevation — the kamestm HIGHEST tier was retired for KAME because
per-record analyses cannot honour its precondition — and the two tiers are not
a matter of degree.  `if(entry_pr == Priority::HIGHEST) break;` sits at the top
of the negotiator's round loop, above both `negotiate_sleep` call sites, so
HIGHEST cannot park; NORMAL parks in 1–2 ms chunks.  Same host, same roles,
120 s, only the tier:

| tier | p50 | p99 | p99.9 | **MAX** |
|---|---|---|---|---|
| HIGHEST | 768 ns | 2.05 µs | 20.5 µs | **95.1 µs** |
| NORMAL (shipped), 20 ms budget | 768 ns | 1.28 µs | 3.67 ms | **20.15 ms** |

Identical median, 200× apart in the tail, and the signature is unambiguous:
the other roles completed a mean of 2,004 commits during each slow NORMAL
commit against 13 in the HIGHEST arm.  `SCHED_FIFO` changed nothing in *this*
measurement (43.8 k/s and MAX 20.15 ms with, 43.3 k/s and 20.19 ms without) —
but the inference published with it, "no scheduling class shortens a voluntary
wait", is **false**; see the correction at the end of this subsection.  Both
arms are budget-dominated at 20 ms, and Linux gives a `SCHED_OTHER` task 50 µs
of timer slack on a futex timeout and an RT task none, so they were never the
same measurement at the scale where the difference lives.

**At NORMAL the wait budget is the only bound the record commit has, and —
isolated — it delivers exactly its value.**  `SCHED_FIFO` + pin, 60 s each:

| budget | commits/s | mean | **MAX** | MAX − budget | clipped |
|---|---|---|---|---|---|
| 2 ms | 76,904 | 7.99 µs | 2.179 ms | 179 µs | 0.333 % |
| 1 ms | 131,721 | 4.10 µs | 1.223 ms | 223 µs | 0.320 % |
| 500 µs | 185,700 | 2.41 µs | 0.662 ms | 162 µs | 0.304 % |
| 200 µs | 251,933 | 1.53 µs | **0.408 ms** | 208 µs | 0.334 % |

MAX = budget + a constant ~200 µs, **no floor down to 200 µs**, and throughput
*rises* 3.3× as the budget falls because a clipped commit stops sleeping and
retries; the clip rate is invariant at ~0.32 %, i.e. the same population caught
earlier and cheaper.  (`XPrimaryDriver::downstreamWaitBudgetUS()`'s documented
"20 ms costs 4.7 %" came from the grand-scope 8-thread arm and does not hold
here.)  Confirmed at length — 300 s, 1 ms budget: **38,303,308 commits, MAX
1.288 ms, zero over a 3 ms deadline**, every other role healthy.

> **The ~200 µs constant's attribution was wrong, and the constant is gone.**
> This subsection credited it to the budget-exempt wait.  Instrumenting the
> negotiator refuted that — `rounds_exempt` is **zero across 17,274 slow
> commits** under every scheduling class, C-state setting and budget — and
> located it in the timed wait's own **wake-up**: scheduling class 5.3×
> dominant, PM-QoS 6× only on top, super-additive, which is why the single-knob
> arms in this document called the residue irreducible.  Budgeted sleeps now
> stop `KAME_NEG_SPIN_TAIL_US` (300 µs) short of the deadline and poll the
> remainder, taking MAX − budget to **7.1 µs at 20 ms and 3.0 µs in the ship
> configuration** — under this host's 17 µs floor.  So the table and the soak
> above are **pre-reserve** data: their shape holds, their constant does not.
> The full entry, including the open item (the reserve is not yet capped to a
> fraction of the budget *span*, so budgets at or below 300 µs starve the
> deferrable tiers) lives in
> [`kamestm/design/RT_READINESS.md`](../../kamestm/design/RT_READINESS.md) —
> this is the allocator's readiness record, and the STM's commit latency is
> quoted here only as the context G7's bands sit in.

**Isolation is what makes the budget work — and this corrects the reading
above.**  §G7's earlier note that isolation is "marginally faster, and nothing
argues against it" was a *throughput* observation at a 20 ms budget.  In the
latency dimension it is not marginal and not optional:

* **Unpinned**, MAX sticks at **12–13 ms for every budget from 5 ms down to
  500 µs** while the clipped count saturates.  The standing explanation is the
  wait behind a **live privileged peer**, which is contractually budget-exempt
  and therefore bounded by the holder's completion time — but this is a
  **hypothesis, not a measurement**: it is the same one the instrumentation
  above refuted for the *pinned* arm, and the instrumented build has not been
  run unpinned.  One `rounds_exempt` reading over an unpinned arm settles it.
  (And "the holder's scheduling delay and nothing else", as this subsection
  first put it, is one term short in any case — a HIGHEST peer's commits
  re-run a privileged holder's closure, which isolation does not touch.  See
  the precondition in `kamestm/README.md`.)
* **Pinned** — acquisition alone on the isolated core, all contenders together
  on the housekeeping core — the holder is always promptly scheduled among its
  peers and the exempt residue vanishes entirely (table above).
* **`SCHED_FIFO` without isolation FAILS the livelock watchdog.**  Only the
  acquisition thread is elevated, so it preempts the very CFS holders it then
  waits behind: UI fell to 144 commits/s and SCRIPTING to 176 (from 42.3 k and
  129.7 k), both flagged at 6,001 ms, while acquisition ran away at 337 k/s and
  still took 50.9 ms on its own worst commit.  A textbook priority inversion.
  **FIFO and isolation ship together or neither ships.**

Refuted along the way, since it was the obvious suspect: the cross-subtree
`XSecondaryDriver` role is not what the 12–13 ms residue is made of — turning
it off halves the clipped population (0.077 % → 0.039 %) and leaves MAX at
12.0 → 13.0 ms.

**Consequence for `raiseAcquisitionOSPriority_()`**, whose Linux implementation
was waiting on exactly this measurement: it must not raise to `SCHED_FIFO`
unless the thread is also isolated from the threads it will negotiate with.
Raising alone is the failing arm above, not a partial win.

### G8 — §74 single mmap+radix site — **DONE, no work remaining**

Already landed in `c04a7975d`: `allocate_chunk<ALLOC>()` no longer carries
its own region walk / mmap / bitmap-CAS — it is LRC-pop → `claim_chunk`
→ header-stamp, and `mmap_new_region()` has exactly one caller
(`claim_chunk`, `allocator.cpp:3957`).  The `(§74) The SINGLE region-walk
+ mmap + bitmap-claim site` note at `allocator_prv.h:1031` is a statement
of the achieved state, not a TODO.  See §1.1 for the resulting four-site
audit surface — G1–G3 gate those directly, with no refactor first.

### G9 — teardown L1-stranding — **DONE, negative control passed on Linux**

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

**The negative control has now been run on Linux, and the test has teeth.**
It was owed because macOS *cannot* trigger the bug — glibc runs C++
`thread_local` destructors before `pthread_key` destructors, which is what
opens the window, and dyld's order does not — so a pass there proved nothing.

Host: Ubuntu 24.04, glibc 2.39, kernel 6.18.5, x86-64, GCC 13.3, Release.

1. **As-is, both linkages pass** (`alloc_thread_exit_unarmed_test` and
   `_dynamic`): `units_live` / `chunks_live` plateau at 10 / 6 from cycle 40
   through cycle 119.
2. **Guard reverted** — dropping `|| kame_thread_torn_down()` from `l1_push`
   and nothing else — **both linkages FAIL**, with exactly the predicted
   signature: `chunks_live` 25 → 125 across 100 measured cycles, i.e. +1
   chunk per cycle, monotonic, no plateau.
3. Guard restored; both pass again; full ctest 18/18.

The ordering assumption was confirmed directly rather than assumed: a probe
registering both a C++ `thread_local` destructor and a `pthread_key`
destructor on the same thread shows the `thread_local` one running **first**
on glibc 2.39.

Worth recording, because it is what makes the guard load-bearing rather than
redundant: instrumenting `l1_push` at the moment the consumer's `pthread_key`
destructor frees the foreign block shows `s_l1_drained == 0` and
`kame_thread_torn_down() == 1`. The consumer never armed its L1 (so the
pre-existing `s_l1_drained` check cannot fire — this is the never-armed case
`3145e139a` was written for), but its allocator TLS *was* armed, so
`AllocThreadExitCleanup`'s `thread_local` destructor had already set
`s_alloc_tls_off`. The `kame_thread_torn_down()` term is therefore the only
one that closes the window, and the test exercises precisely that.

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

**Remaining: none.**  The last two items both needed a Linux host and have
now been done there:

```
G6(a)  DONE — kame_pool_set_thp_policy(), covering the 32 MiB regions AND
       the large-VA tier, re-advising regions already mapped.  Opt-in;
       set_realtime_mode(1) deliberately does NOT imply it.  Numbers in G6(a).
G9-nc  DONE — guard reverted on Linux, alloc_thread_exit_unarmed_test FAILS
       +1 chunk/cycle exactly as predicted; guard restored, 18/18 green.
```

Everything is done and measured on both the same-thread and cross-thread
paths, with the claims and their exclusions written down (G4, G10).  The
Linux-side working notes, including the container traps worth knowing before
re-running any of it, are in `design/RT_LINUX_HANDOFF.md`.
