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
| **Protocol correctness machine-checked** | chunk-claim / recycle / orphan-chain protocols TLA+ + GenMC (RC11) verified (see `tests/`); this doc is about *latency bounds*, not correctness. |

## 2. Gaps (the remaining work), prioritized

### G1 — free() can still madvise/munmap inline, even in RT mode  ★the real hole

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

### G3 — pre-reserve mode (cap is only a ceiling)

`kame_pool_set_memory_cap` bounds growth but does not *commit* memory up
front.  **Work:** a hard mode that mmaps + touches N regions at init and
then never maps again (fail or libc-fallback + counter past it).

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

### G8 — prerequisite refactor: §74 single mmap+radix site

Task #74 (de-dup `allocate_chunk` via `claim_chunk`,
`allocator_prv.h:1031`) is still pending.  Doing it first leaves **one**
audited mmap site, which G1–G3 then gate — materially simplifies both
the implementation and the argument.

### G9 — close the teardown L1-stranding OPEN first

The known narrow OPEN (dedicated/large post-teardown L1 stranding:
`30ea1daa` guards `s_l1_drained` but not `kame_thread_torn_down()`)
lives in the same drain machinery G1/G5 modify.  Fix it before or with
the RT work to avoid churning that code twice.

### G10 — the RT contract, stated in the README

One section: *given* realtime mode + prewarm + stable working set +
bounded threads, malloc/free in [1 B .. 32 MiB] performs no syscalls and
completes in a bounded number of atomic steps; sizes > 32 MiB and
working-set growth are excluded; violations are observable via the G2
counters.

## 3. Suggested order

```
G8 (§74 single-site refactor)  →  G9 (teardown OPEN)
  →  G1 (RT-gate free-path reclaim, + rt_drain)
  →  G2 (prewarm API + violation counters)
  →  G5 (per-op inherited-work cap)
  →  G7 (WCET tail harness — validates G1..G5 empirically)
  →  G4 + G10 (bound theorem + contract docs)
  →  G3 / G6 as the target platform demands
```

Rationale: G8/G9 shrink the surface; G1/G2 remove the actual syscalls
and make violations observable; G5 bounds the amortized-to-worst-case
conversion; G7 provides the numbers; only then are the G4/G10 claims
worth writing down.
