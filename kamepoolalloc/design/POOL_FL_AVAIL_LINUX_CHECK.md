# `claude/pool-fl-avail` (416c76b5e) — Linux validation

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
