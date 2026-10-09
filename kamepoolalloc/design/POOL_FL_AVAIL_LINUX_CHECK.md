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
`MASK_CNT` decrement.

**B. Room not visible to a live owner** (accepted cost, `Inv_NoLostRoom` in
the onebit model).  A freer finds Q set; the owner's `take_rest` drops Q and
`allocate_pooled` fails on the 94 % gate (`:1812`) or FS=false fragmentation;
then the freer's clears land.  With no later free on that chunk it has room
but is on no list, and nothing scans the DLL any more; while the owner lives
only the neighbour release (two chunks after the pin) recovers it.  Plausible
from the code, not observed: §4.4 is where it would show.  Re-checking
`rv_take_q()` after the clears when `q` was false would close it.

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
