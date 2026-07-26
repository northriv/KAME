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

### G4 — WCET bound argument for the lock-free retries (docs/theory)

Lock-free ≠ wait-free.  The unbounded-in-principle loops:

- bitmap word CAS claims, alloc side (`:1373/:1405/:1580`) and free side
  `batch_clear_impl` (`:1962`) — each CAS failure implies another
  thread's *success* on that 64-slot word, so per-word failures are
  bounded by interfering ops, giving an O(T)-style envelope under
  bounded thread count;
- seqlock `lookup_chunk` reader retry — writers are cold-path only
  (chunk claim/release), so reader retries are bounded by cold-path
  frequency, which RT mode + prewarm drives to zero;
- orphan-chain push/pop CAS (`:7371/:7412`) — thread-exit path only.

**Work:** write the per-op bound statement with its assumptions (bounded
T, prewarmed, RT mode) as a documented theorem; optionally add
bounded-retry + per-thread emergency-reserve fallback for the paranoid
hard-RT profile.

### G5 — bound the deferred work a single free() can inherit

Cross-thread dealloc batches (BMWIN multi-word claim) and orphan-chunk
adoption mean one `free()` can inherit accumulated cleanup.  Adoption
pop is O(1)-CAS, but batch drains need an audit: cap the per-op inherited
work (drain at most K words / adopt at most one chunk per call), pushing
the remainder to the next op or to `kame_pool_rt_drain()`.

### G6 — page-fault class (kernel-side latency without syscalls)

- First-touch faults on mapped-but-untouched pages → prewarm must touch
  (G2.1).
- macOS `MADV_FREE`'d pages re-fault with lazy zeroing on reuse — another
  reason G1 must keep pages warm in RT mode.
- Linux THP (`THP=always` hosts): khugepaged/compaction can stall any
  fault — offer `MADV_NOHUGEPAGE` opt-in for RT arenas, or document
  `mlockall` + `vm.compaction_proactiveness` guidance.

### G7 — WCET tail harness (measurement)

Existing benches are throughput-oriented.  RT claims need a per-op
latency harness: RT-priority thread + N interferer threads, per size
band, RT mode on, prewarmed — reporting **max and p99.9999**, not means;
plus the G2 violation counters asserted zero.  CI-able smoke variant.

### G8 — §74 single mmap+radix site — **DONE, no work remaining**

Already landed in `c04a7975d`: `allocate_chunk<ALLOC>()` no longer carries
its own region walk / mmap / bitmap-CAS — it is LRC-pop → `claim_chunk`
→ header-stamp, and `mmap_new_region()` has exactly one caller
(`claim_chunk`, `allocator.cpp:3957`).  The `(§74) The SINGLE region-walk
+ mmap + bitmap-claim site` note at `allocator_prv.h:1031` is a statement
of the achieved state, not a TODO.  See §1.1 for the resulting four-site
audit surface — G1–G3 gate those directly, with no refactor first.

### G9 — teardown L1-stranding — **fix DONE; only a regression test is owed**

Closed by `3145e139a`: both L1 entry points now gate on
`kame_thread_torn_down()` — `l1_push` (beside the pre-existing
`s_l1_drained` check, so never-armed threads are covered too) and
`l1_pop_fit` (no re-arm, closing the symmetric `g_lrc_l1_threads`
counter drift).  Verified in the current tree
(`s_l1_drained || kame_thread_torn_down()` at `allocator.cpp:6803`,
`kame_thread_torn_down()` at `:6770`).

Residual is **test coverage, not a fix**: a manifesting regression test
(cross-thread > 32 KiB block freed from a *non-allocating* thread's
`pthread_key` destructor — the same-thread scenario B does not catch it)
plus Linux verification.  macOS cannot trigger it (dyld runs key dtors
before C++ `thread_local` dtors, unlike glibc), so it belongs to a Linux
run.  Worth doing alongside G7's harness, but it does not block G1.

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

**Remaining:**

```
G5 (per-op inherited-work cap: bound the drain / adoption
    work a single free() can inherit)
  →  G7 (WCET tail harness: max & p99.9999 under interference,
          asserting rt_violations()==0; fold in G9's owed
          manifesting regression test on Linux)
  →  G4 + G10 (CAS-retry bound statement + the RT contract in
                the README, written against G7's numbers)
  →  G6 as the target platform demands (THP / MADV_NOHUGEPAGE,
          mlockall guidance)
```

Rationale: the syscalls are now gated and observable, so what is left is
bounding the *amortized-to-worst-case* conversion (G5), then measuring
(G7) — and only then writing the G4/G10 claims, which should rest on
measured numbers rather than on the design argument alone.
