# `claude/pool-fl-avail` (416c76b5e) — Linux validation

> **On master:** the branch validated here landed as §revive (`39edad5a9`) and
> §group (`bd7875d06`); §6 is the run on that head, §7 on the v1.2.0 code
> (`8dbc7d902`) and §8 on the fix of its regressions (`2e8f940fd`, the v1.2.1
> candidate), §9-§10 the thread-teardown stress tests it was given, §11
> the `kame_pool_rt_drain()` change (`67731ae56`), and §12 calloc's
> fresh-mapping memset skip (`9bb57a41f`) with its test (`19b2e4758`).  This
> record was brought over from `claude/great-turing-Ufao2` (`9e08cb95f`; §7
> and §8 from `7ac5f6bc6`, §9 from `3f7f78934`, §10 from `498413dd1`, §11
> from `604d8454a`, §12 from `093ab4870`).  `kamestm/tests/tmin_dynnode.cpp`,
> cited in §4.3, belongs to the dynnode investigation and stays on that
> branch.

Run 2026-10-09 in the cloud session (x86-64, g++ 15, 4 cores, glibc).
The branch is compared throughout against master 872d89030, built the same
way (`RelWithDebInfo` with `CMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG"`,
`USE_KAME_ALLOCATOR=ON`).  Every A/B interleaves the two arms, one binary per
arm with the pool RPATH-pinned, and is scored by exit status.

The branch: §fl-avail (per-thread lists of chunks with freelist entries),
§revive (anchors, revival stacks, `BIT_Q`; the force-walk hint is gone),
§group (an exiting thread hands over its chunks as one group; the
chunk-wise orphan chain and its scrub retire), plus the serial in the
orphan chain's `atomic_shared_ptr` words.

## Summary

| check | branch | master |
|---|---|---|
| no-DCAS audit (`check_no_dcas.sh`, with gcc-multilib) | 3/3 ok | — |
| LP64 ctest, release | 42/42 | — |
| LP64 ctest, asserts on, 18 vs 15 full runs | 1 `atomic_queue_test` integrity failure (see §2) | 1 `alloc_tsd_exclusivity_test` SIGSEGV (force-walk TOCTOU) |
| ILP32 ctest, i586 and i486, release and asserts on | all pass but `transaction_wait_budget_test`, a load artifact (§3) | — |
| `alloc_tsd_exclusivity_test`, 4-way, 3000/arm | **0 / 3000** | **34 / 3000**, all SIGSEGV |
| thread-exit soak, 4 tests × 200/arm | 0 / 800 | 0 / 800 |
| `tmin_dynnode 100 16 1250`, LP64 | 0 / 12 | 0 / 12 (no baseline signal) |
| `tmin_dynnode 100 16 1250`, ILP32 i586 | 0 / 12 | 0 / 12 (no baseline signal) |
| chunk/region growth (§4.4) | same or fewer chunks | — |

The force-walk TOCTOU (`FORCE_WALK_TOCTOU_HANDOFF.md`) is gone: 0/3000
against 34/3000, Fisher p ≈ 1e-10.  Nothing regressed that the tests here
can see.  A static review (§5) found no TLS-lifetime, double-list or 32-bit
defect; it found one narrow write-after-release window that predates the
branch, one accepted "room not visible" case, and several comments that the
branch has made wrong.

## 1. Builds

- LP64: `/tmp/.../b_fla64` (release), `b_fla64a` (asserts on: `-O2 -g`,
  `-UNDEBUG`); master `b_mb64`, `b_mb64a`.
- ILP32: `-m32 -march=i586` and `-m32 -march=i486`, release and asserts on —
  four trees.  The i486 trees build 36 tests against i586's 43; the rest are
  excluded on a host without a 64-bit CAS, as intended.
- `tools/audit/check_no_dcas.sh`: all three phases ok.  No `__atomic_*_8`
  in the i486 objects of either library.

## 2. The `atomic_queue_test` failure is not the allocator

One asserts-on full ctest of the 18 failed it:

```
test2:failed queue1size=0, queue1total=0, queue2size=0, queue2total=0, queue3size=0, queue3total=50746
```

Oversubscribed A/B (8 concurrent copies on 4 cores, 60 rounds per arm):
**branch 1/480, master 0/480** — no difference.  Only queue3 is off, and
queue3 is `atomic_queue_reserved<int, 3>`, which allocates nothing: values
live in a fixed `m_array`, keys are recycled through `m_reservoir`.  The
key is `index * 0x100 + (serial % 0xff) + 1` — an 8-bit serial — and
`atomicFront()` reads `m_array[idx]` without any atomic, before
`atomicPop(key)` validates the key.  A reader preempted across ~255
recyclings of one of the three slots validates a stale key and subtracts a
value that is not the one popped.  That is an ABA in `kamestm/atomic_queue.h`,
which is identical on master; oversubscription is what makes the preemption
long enough.  Not an allocator finding, and not fixed here.

## 3. `transaction_wait_budget_test` on ILP32

All four ILP32 trees failed it under `ctest -j4` (one of the runs also
overlapped a compile): p99.99 of the 100 µs and 1000 µs arms exceeded
budget + slack, with the unbudgeted arm's own p99.99 at 9.6 ms — i.e. the
host, not the negotiator, was slow.  Alone, interleaved, i586:
**branch 3/3 pass, master 3/3 pass**, p99.99 74–94 µs / 856–881 µs on both.
A 4-thread latency test on a 4-core box under `-j4` measures the box.

## 4. A/B results

### 4.1 Exclusivity (the force-walk TOCTOU)

`alloc_tsd_exclusivity_test`, 4 concurrent per arm-round, 750 rounds:

| arm | nonzero | detail |
|---|---|---|
| pool-fl-avail | **0 / 3000** | |
| master | **34 / 3000** | all exit 139 (SIGSEGV) |

Master's rate (1.1 %) matches the TOCTOU measurement in
`FORCE_WALK_TOCTOU_HANDOFF.md`; the baseline fires, so the 0 is meaningful.

### 4.2 Thread-exit soak

`alloc_thread_exit_free_test`, `..._dynamic`,
`alloc_thread_exit_unarmed_test_dynamic`, `alloc_thread_churn_test`;
4 concurrent, 50 rounds, arms interleaved per test: **0 / 200 each, both
arms**.

### 4.3 `tmin_dynnode`

`kamestm/tests/tmin_dynnode.cpp` (from this branch, `DYNNODE_UAF_HANDOFF.md`
§3), built with `-DA_NO_P1TREE -O3` against each arm's own kamestm headers and
RPATH-pinned to its pool; one run at a time, arms interleaved, 12 rounds each:

| | pool-fl-avail | master |
|---|---|---|
| LP64 | 0 / 12 | 0 / 12 |
| ILP32 i586 | 0 / 12 | 0 / 12 |

**This does not discriminate.**  Master no longer fires it on this box (the
handoff's 40–65 % was on an older base), so the 0 on the branch says only that
the branch did not bring the failure back — it is not evidence that the
branch fixes anything here.

### 4.4 Chunk / region growth

One process at a time, arms interleaved, 3 reps.  Throughput is not
compared — a shared 4-core box is not a benchmark host.

`bench_xthread_pool -w 2 -t 3 -s <size>` — regions added, `chunks_live` at the end:

| size | pool-fl-avail | master |
|---|---|---|
| 64 B | +2 / +2 / +2; 287–297 | +2 / +2 / +2; 282–287 |
| 256 B | +3 / +1 / +1; 101–117 | +1 / +1 / +1; 112–122 |
| 1024 B | +1 / +1 / +1; 102–127 | +1 / +1 / +1; 156–157 |

`alloc_stress_test` (defaults: 2000 threads, 32 concurrent, 20000 ops,
10 % cross-thread): PASS ×3 both arms; VmHWM 194 / 202 / 200 MiB against
211 / 207 / 204 MiB.

No growth.  The branch holds as many or fewer chunks (a third fewer at
1024 B); the single +3 at 256 B did not repeat.  Review finding B (§5), room
a live owner cannot see, would appear here as extra chunks, and none appear —
which bounds it for these workloads but does not rule it out for a
long-lived owner whose chunks receive no later free.

## 5. Static review of the diff

Read-only review of `allocator.cpp`, `allocator_prv.h`,
`atomic_smart_ptr.h` against master, with the RevivalAnchor / RevivalStack /
RevivalGroup specs as intent.  Line numbers are the branch's.

**No defect found** in: pointers into another thread's TLS (none remain —
every chunk link now points at a chunk; `fl_avail` is written only on owner
paths, so the shared `g_teardown_page` is never written); double push / double
pop on the revival stacks and ROOM/FULL chains (Q is taken only when clear and
dropped only by the consumer after it has read `m_rv_next`; heads are emptied
only by exchange; the ROOM pop has the serial, FULL is detached by swap); the
dissolve test (`refcnt == 1`); 32-bit width (18 low bits + 15-bit serial,
every shift < 32, `m_rv_head`/`refcnt`/`m_seen_serial` are `uintptr_t`, no new
64-bit atomic); memory order on the publish/consume edges.

**A. Late write by a freer that did not hold Q** (low severity, predates the
branch).  `return_slots` (`allocator.cpp:2186`, `2357`): a freer that finds Q
already set holds nothing once its last word clear brings `MASK_CNT` to 0, yet
still writes afterwards — FS=false: `atomicDec(&m_flags_filled_cnt)`
(`:2231`); FS=true (and FS=false, whose `batch_clear_impl` is the FS=true
base's): `m_last_coalesce_x16.store` (`:2320`).  In that gap the Q holder can
release the chunk (`group_take_head` CAS from `BIT_OWNED|BIT_Q`, `:8185`;
the exit `settle`), or Q can be dropped and `owner_release` (`:3121`) or the
exit drain (`:3265`) succeed.  Master's `owner_release` has the same window
(pre-check `MASK_CNT == 0`, then `atomicFetchAnd(~BIT_OWNED)`), so this is not
new; the branch adds release sites of the same kind.  Consequence: released
chunks stay mapped (regions are push-only; `exit_release_chunk` at most
madvises), so the write cannot fault; if it lands after a new chunk was
constructed in the same place it sets a relaxed hint byte or leaves
`m_flags_filled_cnt` (an `int`) at −1 — both heuristics, not bitmap state.
The `BIT_Q` comment at `allocator_prv.h:1916-1918` ("nobody releases a chunk a
freer may still be touching") overstates: it holds only for the freer that
holds Q.  Fix if wanted: do the counter and hint writes before the last
`MASK_CNT` decrement.  **Fixed in bd7875d06** that way — see §6.

**B. Room not visible to a live owner** (accepted cost, `Inv_NoLostRoom` in
the onebit model).  A freer finds Q set; the owner's `take_rest` drops Q and
`allocate_pooled` fails on the 94 % gate (`:1812`) or FS=false fragmentation;
then the freer's clears land.  With no later free on that chunk it has room
but is on no list, and nothing scans the DLL any more; while the owner lives
only the neighbour release (two chunks after the pin) recovers it.  Plausible
from the code, not observed: §4.4 is where it would show.

*Retracted suggestion.*  The review proposed re-checking `rv_take_q()` after
the clears when `q` was false.  That is unsafe, as bd7875d06 points out: by
then no slot of the freer's keeps the chunk alive, so the CAS can land on a
chunk already released and rebuilt — finding A's window again, with an atomic
write in it instead of a counter.  `RevivalStack.tla` with `NoPin` violates
`Inv_NoUseAfterRelease`.  B stays an accepted cost of the one-bit protocol.

**Comments the branch makes wrong** (most were stale before, but §group makes
them misleading):
- `allocator.cpp:2979` — "an anchor is never listed"; anchors are listed on
  their own head.
- `allocator.cpp:2434` (§35) — `batch_return_to_bitmap` "releases the chunk";
  freers never release now.
- `allocator_prv.h:1858-1885`, `allocator.cpp:3076-3100` — the `owner_release`
  / `cross_release` / `release_dll_chunks_for_thread` docs still describe
  `BIT_RELEASED` / `BIT_OWNER_EXITED`.
- `allocator.cpp:3136-3149` — the `cross_release` stub calls a cross-thread
  decrement to 0 the release path, which §group rules out.
- `allocator.cpp:2311-2316` — "skip for FS=false"; FS=false chunks run the
  FS=true base's `batch_clear_impl`, so they do store.

## 6. bd7875d06 (finding A fixed, comments corrected)

The FS=false `OnClearFn` now decrements `m_flags_filled_cnt` before
`MASK_CNT`; FS=true already did.  The coalescing hint is stored inside
`batch_clear_impl`'s loop just before the last word's clear, which is the only
clear that can bring `MASK_CNT` to 0 — the freer's own bits keep every later
word live until then.  No caller of `batch_return_to_bitmap` / `return_slots`
touches the chunk after it returns (`flush` advances over its own buffer;
`push_direct` and both teardown bypasses return at once).  Read and agreed.

Same trees rebuilt from bd7875d06, same method as above:

| check | result |
|---|---|
| no-DCAS audit | 3/3 ok |
| ctest LP64 release / asserts | 43/43, 43/43 |
| ctest ILP32 i486 release / asserts | 36/36, 36/36 |
| ctest ILP32 i586 release / asserts | 42/43 each: `transaction_wait_budget_test` again under `ctest -j3`; alone 6/6 PASS (3 per tree) — the §3 load artifact |
| `alloc_tsd_exclusivity_test`, 2000/arm | **0 / 2000** vs master **34 / 2000** (all SIGSEGV) |
| thread-exit soak, 4 tests × 80/arm | 0 / 320 vs master 0 / 320 |

Nothing changed for the worse; the TOCTOU result holds on the new head.

## 7. master 8dbc7d902 (the v1.2.0 candidate)

The same set as §6, on master after §revive / §group landed and the
sanitizer work followed: d61f792be, 23d843e23 and d6cc5007d (relaxed atomics
where bitmap words, filled counts, slot headers and `m_owner_id` were
accessed plainly — TSan), d27a065d9 (`m_owner_id` read and stamped by offset
— UBSan vptr), ae2603fbc (kamestm `Payload` keeps a `Node<XN>` pointer — UBSan
vptr), the three UBSan fixes (c34ae60f0, 471a34981, 82ff63d5e), and
`atomic_queue_reserved` in atomics (5d3a9ce30).  Fresh worktree, the same six
build configurations, the same baseline (old master 872d89030) for the A/Bs.

| check | result |
|---|---|
| builds, six trees | all clean, no undefined reference |
| no-DCAS audit | 3/3 ok — the 64-bit slot-header store (23d843e23) is split into two 32-bit halves on 32-bit hosts, as the probe confirms |
| ctest LP64 release / asserts | 43/43, 43/43 |
| ctest ILP32 i586 release / asserts | 43/43, 43/43 (`transaction_wait_budget_test` happened to pass under `-j3` this time) |
| ctest ILP32 i486 release / asserts | 36/36, 36/36 |
| `alloc_tsd_exclusivity_test`, 2000/arm | **0 / 2000** vs old master **22 / 2000** (all SIGSEGV) |
| thread-exit soak, 4 tests × 80/arm | 0 / 320 vs old master 0 / 320 |
| LP64 GCC UBSan (pool on), whole tree | **0** `runtime error` lines (old master: 141 — 137 vptr at four `allocator.cpp` sites, 4 invalid-bool in `walkUpChain`) |

UBSan's ctest: 41/43.  `transaction_wait_budget_test` is the load artifact.
`transaction_nosyscall_highest_test` segfaults, now with no diagnostic at
all: the first vptr type-cache miss on the seccomp-filtered thread (in
`std::thread::_State_impl::~_State_impl`, a slow-path check, not a report)
lazily runs `__ubsan::InitAsStandalone()`, whose read of
`/proc/self/cmdline` the filter traps; `internal_strncpy(src = 0x9)` faults.
The sanitizer runtime making syscalls on the filtered thread is exactly what
the test's CMake note rules out ("NOT sanitizer-compatible, by
construction"); without UBSan it passes in all six trees.

Nothing regressed; the TOCTOU result holds; the UBSan vptr sites are gone.

## 8. master 15a1dc312 (2e8f940fd, the v1.2 regression fix)

The Linux validation `V1_2_PERF_HANDOFF.md` lists as open: 2e8f940fd
(§fl-scan / §walk, the `FL_UNTRACKED` marker) had only run on macOS, where the
no-DCAS i486 probes skip.  The §7 set again, fresh worktree, plus the §4.4
growth check against 8dbc7d902 (the v1.2.0 code) instead of old master.

| check | result |
|---|---|
| builds, six trees | all clean, no undefined reference |
| no-DCAS audit | **3/3 ok** (the open item) |
| ctest LP64 release / asserts | 43/43, 43/43 |
| ctest ILP32 i586 release / asserts | 43/43, 43/43 |
| ctest ILP32 i486 release / asserts | 36/36, 36/36 |
| `alloc_tsd_exclusivity_test`, 2000/arm | **0 / 2000** vs old master **16 / 2000** (all SIGSEGV) |
| thread-exit soak, 4 tests × 80/arm | 0 / 320 vs old master 0 / 320 |
| LP64 GCC UBSan (pool on), whole tree | **0** `runtime error` lines; 41/43, the same two as §7 (`wait_budget` under load, `nosyscall_highest` vs the UBSan runtime) |

Growth, 15a1dc312 against 8dbc7d902, one process at a time, interleaved, 3
reps — `bench_xthread_pool -w 2 -t 3 -s <size>`, regions added and
`chunks_live` at the end:

| size | 15a1dc312 | 8dbc7d902 |
|---|---|---|
| 64 B | +2 / +2 / +2; 290–296 | +2 / +2 / +2; 291–299 |
| 256 B | +1 / +4 / +4; 104–146 | +3 / +5 / +3; 127–140 |
| 1024 B | +1 / +1 / +3; 106–129 | +3 / +2 / +1; 102–137 |

`alloc_stress_test` (defaults): PASS ×3 both; VmHWM 197 / 200 / 193 MiB
against 194 / 213 / 191 MiB.  No growth from the fix; the 256 B region counts
are noisy on this box in both arms alike.

Not run here: the `bench_ab.sh` performance A/B.  This is a shared 4-vCPU VM,
not a benchmark host; the handoff's x86-64 reference is Ohtaka.

## 9. `alloc_tsd_exclusivity_var_test` (master b775f5335, the v1.2.1 candidate)

b775f5335 adds the FS=false sibling of `alloc_tsd_exclusivity_test`: the same
teardown shape over 369..32768 B (borrow tier at ALIGN 64 / 256, full-usable
at 1024, page-sized and bucket 51 at 4096), with a paint on every block.  Run
2026-10-11, fresh container (x86-64, g++ 13.3, 4 vCPU), the six §8 build
configurations.  Old master is 872d89030 with only the test file and its three
CMake lines added (`-O3 -g -DNDEBUG`, `-g` for addr2line; the code is the same
as the baseline's).  Every arm ran with an `LD_PRELOAD` SIGSEGV reporter (frame
module + offset, then re-raise, so the exit status stays 139), arms interleaved
round by round, 4 processes at a time.

| check | result |
|---|---|
| builds, six trees | all clean, the new test included |
| no-DCAS audit | 3/3 ok |
| ctest LP64 / i586, release and asserts | 44/44 ×4 |
| ctest i486, release and asserts | 37/37 ×2 |

| run | HEAD b775f5335 | old master 872d89030 |
|---|---|---|
| var test, LP64 release, 2000/arm | **0 / 2000** | **65 / 2000**, all SIGSEGV (p = 3 × 10⁻²⁰) |
| 32 B test, LP64 release, 2000/arm | **0 / 2000** | **12 / 2000**, all SIGSEGV (p = 5 × 10⁻⁴) |
| var test, i586 / i486 release, 500 each | 0 / 500, 0 / 500 | — |
| 32 B test, i586 / i486 release, 500 each | 0 / 500, 0 / 500 | — |
| var test, asserts on (LP64 / i586 / i486), 100 each | 0 / 100 ×3 | — |

Where old master falls — the faulting PC, resolved through its inline chain:

| test | site | count | classes |
|---|---|---|---|
| var test | `allocator.cpp:2010`, FS=false `deallocate_pooled`, `cached_force_walk->store(true)` | 65 / 65 | ALIGN 256: 29, 1024: 19, 4096: 9, 64: 8 |
| 32 B test | `allocator.cpp:878`, `CrossDeallocBatch::flush` (via `deallocate_pooled_static<32>`) | 12 / 12 | ALIGN 32 |

Every var-test crash is a cross-thread free of a block taken from the handoff
(test line 201, `tracked_free(taken)`) — the block of a thread that may have
exited, so the store lands in its dead TLS.  The two tests therefore cover
**different** force-walk sites, each one exclusively.

**`push_direct` is reached by neither.**  Its store (`allocator.cpp:815`) is
on the FS=true free path for ALIGN > 48 (`ALIGN <= 48` goes `push` →
`flush`); the var test allocates only FS=false sizes, which take the
FS=false direct site above.  A scratch copy of the var test with
`pick_size` changed to 49..368 B (not committed), against the same two
libraries, 1000/arm:

| | HEAD | old master |
|---|---|---|
| 49..368 B variant | **0 / 1000** | **21 / 1000** (p = 9 × 10⁻⁷): 20 SIGSEGV + 1 timeout |

Of the 20: **7 at `push_direct<ALIGN>`, `allocator.cpp:815`** (ALIGN 80, 144,
208, 304, 336, 352, FS=true), 13 at the FS=false store `:2010` (ALIGN 32 and
256 FS=false classes — 49..368 B spans both).  The timeout (exit 124, nothing
printed) is not classified.  So all three sites are live on the old code and
closed on HEAD, but only two of them are covered by the committed tests; a
49..368 B tier in the var test (or a third variant) would cover the third.
Not changed here.

**Since (on master):** the 49..368 B tier is in the committed test, as its
`--mid` mode (registered as `alloc_tsd_exclusivity_mid_test`), so all three
force-walk sites are now reached by a committed test.  On the M3: HEAD 0/1000;
the three faults injected into 2e8f940fd's new state (exit drain keeps a
freelist, a marked head's `next` left unfixed, a marked cell aimed at) are
caught 20/20 each in either mode.  Its Linux run, old code included, is §10.

## 10. `--mid` (`alloc_tsd_exclusivity_mid_test`, master 3d7fc84e2)

The §9 method again, for the `--mid` mode 3d7fc84e2 added (49..368 B: the
FS=true classes above 48 B, whose cross-thread free takes `push_direct`, and
the small FS=false classes).  Same container and trees as §9, rebuilt at
3d7fc84e2; old master is 872d89030 with the 3d7fc84e2 test file in place of
§9's.  All six trees rebuild clean, and `ctest -R alloc_tsd_exclusivity` runs
3/3 in each (the 32 B, var and mid tests).

| run | HEAD 3d7fc84e2 | old master 872d89030 |
|---|---|---|
| `--mid`, LP64 release, 2000/arm | **0 / 2000** | **32 / 2000**, all SIGSEGV (p = 4 × 10⁻¹⁰) |
| `--mid`, i586 / i486 release, 500 each | 0 / 500, 0 / 500 | — |
| `--mid`, asserts on (LP64 / i586 / i486), 100 each | 0 / 100 ×3 | — |

Where old master falls (all 32 have a backtrace; no timeout this time):

| site | count | classes |
|---|---|---|
| **`allocator.cpp:815`, `CrossDeallocBatch::push_direct<ALIGN>`** (inlined into FS=true `deallocate_pooled_static`) | **15** | ALIGN 80, 112, 304, 320, 336, 368: 2 each; 64, 208, 288: 1 each |
| `allocator.cpp:2010`, FS=false `deallocate_pooled` | 17 | ALIGN 32 (FS=false): 13, ALIGN 256: 4 |

Every one is the cross-thread free of a handed-off block (test line 214,
`tracked_free(taken)`), the store landing in the TLS of an owner that may have
exited.  With §9 the three force-walk sites of the old code each have a
committed test that reaches them on Linux: `flush` (`:878`) the 32 B test,
the FS=false direct store (`:2010`) the var test and `--mid`, and
`push_direct` (`:815`) `--mid` alone.  HEAD is 0 throughout.

## 11. master 67731ae56 (`kame_pool_rt_drain()` keeps the prewarmed caches)

67731ae56 splits the old `kame_pool_rt_drain()` in two: `rt_drain` now settles
only the parked large-tier unmaps, and the new `kame_pool_trim()` also empties
this thread's L1 and the global L2 — where `kame_pool_prewarm()` keeps the
working set above the chunk pool — so a realtime section after a per-cycle
drain no longer maps again.  `bench_rt_wcet` now drains every iteration and
fails on forced releases and on any mapping beyond the band's expected ones.
Same container and six trees as §9–§10, rebuilt at 67731ae56.

| check | result |
|---|---|
| builds, six trees | all clean |
| no-DCAS audit | 3/3 ok (`kame_pool_trim` adds no 64-bit atomic) |
| ctest LP64 release / asserts | 45/45, 45/45 |
| ctest i486 release / asserts | 38/38, 38/38 |
| ctest i586 release / asserts | 44/45 each: `transaction_wait_budget_test` under `ctest -j3`; alone 6/6 PASS (3 per tree) — the §3 load artifact |

The fix, A/B'd with **one** `bench_rt_wcet` binary per tree (built at
67731ae56, which calls only `rt_drain`) against its own library and against the
3d7fc84e2 library (`LD_LIBRARY_PATH` over its RUNPATH; `ldd` recorded in each
log), interleaved:

| run | 67731ae56 library | 3d7fc84e2 library |
|---|---|---|
| default, LP64 / i586 / i486, 3 each | **rt_violations=0**, PASSED ×9 | **rt_violations=1600**, FAILED ×9 ("the realtime arm made 1600 new mapping(s) despite prewarming") |
| `--rt-os-policy 3`, LP64 | PASSED, 0 | **SIGABRT** (exit 134): "RT violation — large-tier mmap from a realtime thread (KAME_RT_OS_ABORT)" |
| `--pressure`, LP64 | PASSED: 800 mappings, all expected (the > LRC_HI band cannot be cached); deferred_unmaps=800, forced_releases=0 | FAILED: 2400 = the 800 expected + 1600 |

Every number matches the commit's (1,600 violations with the per-cycle drain;
800 of 800 large frees deferred, none forced).  A latency comparison is not
made here — a shared 4-vCPU VM is not a host for one.

## 12. master 19b2e4758 (`alloc_calloc_zero_test`; calloc's fresh-mapping memset skip)

9bb57a41f makes `kame_calloc` skip the memset of a fresh §19 mapping
(`allocate_large_va` reports one through a new `zeroed` out-parameter); a
recycled region still gets the memset.  On Linux, where `calloc` was always
interposed, that is the one behavioural change.  19b2e4758 adds
`alloc_calloc_zero_test`: every tier, fresh and recycled (fill 0xA5, free,
calloc again), through `kame_pool_*` and through plain `calloc` / `malloc` /
`free`, with warm reuse asserted in the cached large tier.  Same container and
six trees as §9–§11, rebuilt at 19b2e4758.

| check | result |
|---|---|
| builds, six trees | all clean, the new test included |
| no-DCAS audit | 3/3 ok |
| ctest LP64 release / asserts | 46/46, 46/46 |
| ctest i486 release / asserts | 39/39, 39/39 |
| ctest i586 release / asserts | 45/46 each: `transaction_wait_budget_test` under `ctest -j3`; alone 6/6 PASS (3 per tree) — the §3 load artifact |
| `alloc_calloc_zero_test`, 50 more runs per tree (all six) | **0 / 300** failed |

What the test reports here (LP64 and i486 alike): every case `[ok]`; the
plain-libc pass reaches the pool (`libc ...` rows, as on the `kame_pool_*`
pass); every recycled block came back in every tier up to the 100 MiB large
case (`reused`), so the memset branch ran where it must; the 257 MiB case is
the uncached huge tier.  `malloc_intercept_test` (calloc and the aligned pair
following malloc) passes in all six trees within the ctest counts above.
